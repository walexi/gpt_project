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
define("text_length", default=5, help="length of words generated")


class ModelHandler(object):
	
	def __init__(self):
		self.cache = ""
		self.threshold = options.threshold # default
		self.generated_text = ""
		self.results = ""
		self.file_name = "training_data.txt"
		self.sess = None
		self.initializeModel()

	def initializeModel(self):
		if not os.path.isdir(os.path.join("models", options.model_name)):
			if options.debug:
				print(f"Downloading {options.model_name} model...")

			gpt2.download_gpt2(model_name=options.model_name)   # model is saved into current directory under /models/124M/


		if not os.path.isfile(self.file_name):

			url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
			data = requests.get(url)
		
			with open(self.file_name, 'w') as f:
				f.write(data.text)

		self.sess = gpt2.start_tf_sess()

		if not os.path.isdir(os.path.join("checkpoint", "run1")):
			gpt2.finetune(self.sess, self.file_name, model_name=options.model_name,steps=10)
		else:
			gpt2.load_gpt2(self.sess)

	async def train_model(self, steps):
		if os.path.isfile(self.file_name):
			file_size = os.path.getsize(self.file_name)

			if file_size> 80 * 1000000:
				temp = 'encoded%s.npz'%str(random())
				await gpt2.encode_dataset(self.file_name, out_path=temp)
				await gpt2.finetune(self.sess, temp, overwrite=True, mode_name=options.model_name, steps=steps)
			else:
				await gpt2.finetune(self.sess, self.file_name, overwrite=True, mode_name=options.model_name, steps=steps)

		self.sess = gpt2.start_tf_sess()
		gpt2.load_gpt2(sess)
	
	async def generate_text(self, prefix, text_length=options.text_length):
		gen_text = gpt2.generate(self.sess, length=text_length,
		 prefix=prefix, nsamples=options.nsamples, 
		 batch_size=options.batch_size, return_as_list=True)[0]

		self.results=gen_text;

	async def add_message(self, prefix):
		self.cache+=prefix
		if(len(self.cache)>self.threshold):
			await self.generate_text(self.cache)




model_handler = ModelHandler()
model_handler.threshold = options.threshold
# test = 0
class Generator(tornado.websocket.WebSocketHandler):

	def check_origin(self, origin):
		return True
	def open(self):
		self.model_handler_gen = copy.copy(model_handler)
	async def on_message(self, message):
		await self.model_handler_gen.add_message(message)

		self.write_message(self.model_handler_gen.results)

class Trainer(tornado.web.RequestHandler):
	def initialize(self):
		self.model_handler = ModelHandler()
	async def post(self):
		file = self.request.files['training_data'][0]
		steps = self.request['steps']
		output_file = open(self.model_handler.file_name, 'wb')
		output_file.write(file['body'])

		if not steps:
			await self.model_handler.finetune(steps)
		else:
			await self.model_handler.finetune()

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