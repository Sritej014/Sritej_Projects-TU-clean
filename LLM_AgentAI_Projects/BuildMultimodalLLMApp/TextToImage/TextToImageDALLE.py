from openai import OpenAI
from IPython import display

client = OpenAI()
response = client.images.generate(
    model = "dall-e-2", prompt = "a white mouse",
    size = "1024 x 1024", quality = "standard"
)
url = response.data[0].url
display.Image(url=url, width= 512)