from PIL import Image

source = Image.open("C:\\Users\\chsjk\\Desktop\\images\\butterfly_ori.png")
target = source.resize((3840, 3840))
target.save("test.png")