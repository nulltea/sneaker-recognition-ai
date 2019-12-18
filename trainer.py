from dataset import *;
from config import *;
from model import *;


if __name__ == "__main__":
	dataset = SneakersDataset(IMG_DIR, MODELS);
	net = Net(MODEL_SAVE_PATH);
	print("Init training protocol? (y/n)");
	if input() == "y":
		net.train(data_set=dataset);
