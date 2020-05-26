import tornado.ioloop
import tornado.locks
import tornado.web
import os.path
from tornado.options import define, options,parse_command_line
import gpt_2_simple as gpt2
import requests

define("port", default=8880, help="run the server on the given port")
define("debug", default=True, help="run in debug mode")
define("max_buffer", default=250, help="max buffer size")
define("model_name", default='124M', help="name of pretrained model to load")




class MessageBuffer(object):
	def __init__(self):
		self.cond = tornado.locks.Condition()
		self.cache = []
		self.cache_size = options.max_buffer
		self.results = []
		self.sess = None

	def get_messages_since(self, cursor):
		results=[]
		for msg in reversed(self.cache):
			if msg["id"]== cursor:
				break
			results.append(msg)
		results.reverse()
		return results

	def predict(self, message):
		#check if session is None
		gen_text = gpt2.generate(self.sess, length=250,
		 prefix=message, nsamples=5, 
		 batch_size=5, return_as_list=True)[0]

		return gen_text


	def add_message(self, message):

		self.cache.append(message)
		
		if len(self.cache)>self.cache_size:
			self.cache = self.cache[~self.cache_size:]
			result = self.predict(self, self.cache)
			self.results.append(result)

		self.cond.notify_all()

	




class Generator(tornado.web.RequestHandler):
	def post(self):
		text = self.get_argument("text")
		sess = self.initializeModel()
		res = gpt2.generate(sess, prefix=text, return_as_list=True)[0]
		self.write(res)
	def initializeModel(self):
		if not os.path.isdir(os.path.join("models", options.model_name)):
			print(f"Downloading {options.model_name} model...")
			gpt2.download_gpt2(model_name=options.model_name)   # model is saved into current directory under /models/124M/
		file_name = "shakespeare.txt"
		if not os.path.isfile(file_name):
			url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
			data = requests.get(url)
		
			with open(file_name, 'w') as f:
				f.write(data.text)

		if not os.path.isdir(os.path.join("checkpoint", "run1")):
			gpt2.finetune(sess, file_name, model_name=options.model_name,steps=10)
		else:
			sess = gpt2.start_tf_sess()
			gpt2.load_gpt2(sess)

		return sess
	

class GeneratorUpdates():
	# sess = initializeModel()
	def post(self):
		self.write('')

class Trainer():
	def post(self):
		self.write('')



def main():
	parse_command_line()
	app = tornado.web.Application([
			(r"/generate", Generator),
			(r"/results", GeneratorUpdates),
			(r"/train", Trainer)
		],
		cookie_secret = "generate random value here",
		static_path = os.path.join(os.path.dirname(__file__), "static"),
		xsrf_cookies=False,
		debug=options.debug,
		)
	app.listen(options.port)
	tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
	main()