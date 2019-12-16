from torch.utils.data import Dataset;
from PIL import Image;

CLASSES = ('AIRMAX_270', 'AIRMAX_720', 'AIRMAX_95', 'AIRFORCE1', 'JORDAN', 'REACT', 'VAPORMAX');

class DataSetCreator:
	def __call__(self):
		number_of_batches = len(CLASSES)/ batch_size
		for i in range(number_of_batches):
			batch_x = names[i*batch_size:i*batch_size+batch_size]
			batch_y = labels[i*batch_size:i*batch_size+batch_size]
			batch_image_data = np.empty([batch_size, image_height, image_width, image_depth], dtype=np.int)
			for ix in range(len(batch_x)):
				f = batch_x[ix]
				batch_image_data[ix] = np.array(Image.open(data_dir+f))
			sess.run(train_op, feed_dict={xs:batch_image_data, ys:batch_y})

class NumbersDataset(Dataset):
    def __init__(self, low, high):
        self.samples = list(range(low, high))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = NumbersDataset(2821, 8295)
    print(len(dataset))
    print(dataset[100])
    print(dataset[122:361])