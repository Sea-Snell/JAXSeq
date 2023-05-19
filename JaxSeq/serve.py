from typing import Generator, Any, Optional
import redis
import time
import pickle as pkl
import multiprocessing as mp
from functools import partial
import six
import json
from collections import OrderedDict

# config for server

class Config:
    redis_host = 'localhost'
    redis_port = 6379
    redis_db = 0
    client_refresh_delay = 0.01
    self_indicator = '__self__'
    init_message = '__init_message__'
    sse_channel_prefix = "__sse_channel__"
    sse_exit_type = "__EXIT__"

"""
=====
Below is the code for running a class on a seperate process.
Each call to a method on the class is executed in a queue.
You want to do this when serving models to process 1 request at a time.
=====
"""

def serve_class(cls):
    cache_cls = pkl.dumps(cls)
    
    class WrappedModel:
        def __init__(self, *args, **kwargs):
            self.r = redis.Redis(host=Config.redis_host, port=Config.redis_port, db=Config.redis_db)
            self.Q = initalize_server(self, super().__getattribute__('r'), cache_cls, args, kwargs)
        
        def __getattribute__(self, name):
            return partial(build_method(name, super().__getattribute__('r'), super().__getattribute__('Q')), self)
    
        def __call__(self, *args, **kwargs):
            return build_method('__call__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, *args, **kwargs)
        
        def __getitem__(self, key):
            return build_method('__getitem__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key)
        
        def __setitem__(self, key, value):
            return build_method('__setitem__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key, value)
        
        def __contains__(self, key):
            return build_method('__contains__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self, key)
        
        def __len__(self):
            return build_method('__len__',  super().__getattribute__('r'), super().__getattribute__('Q'))(self)
    
    return WrappedModel

def build_method(method, r, Q):
    def call_method(self, *args, **kwargs):
        request_id = int(r.incr('request_id_counter'))
        Q.put((request_id, method, args, kwargs,))
        while not r.exists(f'result_{request_id}'):
            time.sleep(Config.client_refresh_delay)
        result = pkl.loads(r.get(f'result_{request_id}'))
        r.delete(f'result_{request_id}')
        if result == Config.self_indicator:
            return self
        return result
    return call_method

def server_process(Q, cls_pkl, args, kwargs):
    r = redis.Redis(host=Config.redis_host, port=Config.redis_port, db=Config.redis_db)
    model = pkl.loads(cls_pkl)(*args, **kwargs)
    while True:
        try:
            request_id, method, args, kwargs = Q.get()
            if method == Config.init_message:
                r.set(f'result_{request_id}', pkl.dumps(method))
                continue
            result = getattr(model, method)(*args, **kwargs)
            if isinstance(result, Generator):
                result = tuple(result)
            if result == model:
                result = Config.self_indicator
            r.set(f'result_{request_id}', pkl.dumps(result))
        except EOFError:
            return
        except Exception as e:
            raise Exception

def initalize_server(self, r, cls_pkl, args, kwargs):
    Q = mp.Manager().Queue()
    p = mp.Process(target=server_process, args=(Q, cls_pkl, args, kwargs))
    p.start()
    build_method(Config.init_message, r, Q)(self)
    return Q


"""
=====
Below is the code for handling the server-sent events
=====
"""

# adapted from: https://github.com/singingwolfboy/flask-sse

@six.python_2_unicode_compatible
class Message(object):
    """
    Data that is published as a server-sent event.
    """
    def __init__(self, data, type=None, id=None, retry=None):
        """
        Create a server-sent event.

        :param data: The event data. If it is not a string, it will be
            serialized to JSON using the Flask application's
            :class:`~flask.json.JSONEncoder`.
        :param type: An optional event type.
        :param id: An optional event ID.
        :param retry: An optional integer, to specify the reconnect time for
            disconnected clients of this stream.
        """
        self.data = data
        self.type = type
        self.id = id
        self.retry = retry

    def to_dict(self):
        """
        Serialize this object to a minimal dictionary, for storing in Redis.
        """
        # data is required, all others are optional
        d = {"data": self.data}
        if self.type:
            d["type"] = self.type
        if self.id:
            d["id"] = self.id
        if self.retry:
            d["retry"] = self.retry
        return d

    def __str__(self):
        """
        Serialize this object to a string, according to the `server-sent events
        specification <https://www.w3.org/TR/eventsource/>`_.
        """
        if isinstance(self.data, six.string_types):
            data = self.data
        else:
            data = json.dumps(self.data)
        lines = ["data:{value}".format(value=line) for line in data.splitlines()]
        if self.type:
            lines.insert(0, "event:{value}".format(value=self.type))
        if self.id:
            lines.append("id:{value}".format(value=self.id))
        if self.retry:
            lines.append("retry:{value}".format(value=self.retry))
        return "\n".join(lines) + "\n\n"

    def __repr__(self):
        kwargs = OrderedDict()
        if self.type:
            kwargs["type"] = self.type
        if self.id:
            kwargs["id"] = self.id
        if self.retry:
            kwargs["retry"] = self.retry
        kwargs_repr = "".join(
            ", {key}={value!r}".format(key=key, value=value)
            for key, value in kwargs.items()
        )
        return "{classname}({data!r}{kwargs})".format(
            classname=self.__class__.__name__,
            data=self.data,
            kwargs=kwargs_repr,
        )

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self.data == other.data and
            self.type == other.type and
            self.id == other.id and
            self.retry == other.retry
        )

class SSEServer:
    def __init__(self):
        self.r = redis.Redis(host=Config.redis_host, port=Config.redis_port, db=Config.redis_db)

    def publish(
        self, 
        data: Any, 
        channel: str, 
        type: Optional[str]=None, 
        id: Optional[str]=None, 
        retry: Optional[int]=None, 
    ):
        channel = f"{Config.sse_channel_prefix}/{channel}"

        message = Message(data, type=type, id=id, retry=retry)
        msg_json = json.dumps(message.to_dict())
        return self.r.publish(channel=channel, message=msg_json)
    
    def listen(
        self, 
        channel: str, 
    ):
        channel = f"{Config.sse_channel_prefix}/{channel}"

        p = self.r.pubsub()
        p.subscribe(channel)
        try:
            for message in p.listen():
                if message['type'] == 'message':
                    msg_dict = json.loads(message['data'])
                    if msg_dict['type'] == Config.sse_exit_type:
                        break
                    yield str(Message(**msg_dict))
        finally:
            try:
                p.unsubscribe(channel)
            except ConnectionError:
                pass

