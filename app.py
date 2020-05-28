import copy
import tornado.ioloop
import tornado.locks
import tornado.web
import tornado.websocket
import os.path
from tornado.options import define, options,parse_command_line
import tornado.gen as gen
import gpt_2_simple as gpt2
import requests

define("port", default=8880, help="run the server on the given port")
define("debug", default=True, help="run in debug mode")
define("max_buffer", default=250, help="max buffer size")
define("model_name", default='124M', help="name of pretrained model to load")
define("batch_size", default=5, help="batch size to generate text with gpt2")
define("nsamples", default=5, help="num of samples parameters as defined in gpt2 doc")
define("threshold", default=7, help="length of words before model generates text")


class ModelHandler(object):
	
	def __init__(self):
		self.initializeModel()
		self.cache = ""
		self.threshold = 7 # default
		self.generated_text = ""
		self.results = ""
		self.file_name = "training_data.txt"
		self.sess = None

	def initializeModel(self):
		if not os.path.isdir(os.path.join("models", options.model_name)):
			if optios.debug:
				print(f"Downloading {options.model_name} model...")

			gpt2.download_gpt2(model_name=options.model_name)   # model is saved into current directory under /models/124M/


		if not os.path.isfile(self.file_name):

			url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
			data = requests.get(url)
		
			with open(file_name, 'w') as f:
				f.write(data.text)

		if not os.path.isdir(os.path.join("checkpoint", "run1")):
			gpt2.finetune(sess, file_name, model_name=options.model_name,steps=100)
		else:
			self.sess = gpt2.start_tf_sess()
			gpt2.load_gpt2(self.sess)

	@gen.coroutine
	def train_model(self, steps):
		if os.path.isfile(self.file_name):
			file_size = os.path.getsize(self.file_name)

			if file_size> 80 * 1000000:
				temp = 'encoded%s.npz'%str(random())
				yield gpt2.encode_dataset(self.file_name, out_path=temp)
				yield gpt2.finetune(self.sess, temp, overwrite=True, mode_name=options.model_name, steps=steps)
			else:
				yield gpt2.finetune(self.sess, self.file_name, overwrite=True, mode_name=options.model_name, steps=steps)

		self.sess = gpt2.start_tf_sess()
		gpt2.load_gpt2(sess)

	def generate_text(self, prefix, text_length=50):
		gen_text = gpt2.generate(self.sess, length=text_length,
		 prefix=prefix, nsamples=options.nsamples, 
		 batch_size=options.batch_size, return_as_list=True)[0]

		self.results+=gen_text;

	def add_message(self, prefix):
		self.cache+=prefix
		if(len(self.cache)>self.threshold):
			self.generate_text(self.cache)




model_handler = ModelHandler()
model_handler.threshold = options.threshold

class Generator(tornado.websocket.WebSocketHandler):
	handler_copy = copy.deepcopy(model_handler)

	def check_origin(self, origin):
		return True

	def on_message(self, message):
		handler_copy.add_message(message)

		self.write_message(handler_copy.results)

class Trainer(tornado.web.RequestHandler):
	def post(self):
		file = self.request.files['training_data'][0]
		steps = self.request['steps']
		output_file = open(model_handler.file_name, 'wb')
		output_file.write(file['body'])

		if not steps:
			model_handler.finetune(steps)
		else:
			model_handler.finetune()

		self.finish("Data successfully uploaded and training has started")



def main():
	parse_command_line()
	app = tornado.web.Application([
			(r"/generator", Generator),
			(r"/finetune", Trainer)
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