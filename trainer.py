from dataset import *;
from config import *;
from model import *;
from torch.utils.tensorboard import SummaryWriter
	


if __name__ == "__main__":
	dataset = SneakersDataset(IMG_DIR, MODELS);
	net = Net(MODEL_SAVE_PATH);
	#net.load_model();
	print("Init training protocol? (y/n)");
	if input() == "y":
		net.train(data_set=dataset);