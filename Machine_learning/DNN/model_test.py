import torch
from torch.utils.data import DataLoader

from model_make import *

load_model = torch.load('DNN.pt')

file = ''.join(['data_file'])
dump_data = pickle_data(READ, file, None)

transform = transforms.Compose([
    transforms.Resize((RESIZE_X, RESIZE_Y)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

for idx, value in enumerate(dump_data):
    print('label', idx, 'key', value)
    if idx == 11:
        print('len', len(dump_data[value]))
        for ii, xy in enumerate(dump_data[value]):
            if ii == 100:
                img_array = np.zeros((CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT), dtype=np.uint8)
                img = Image.fromarray(img_array)
                draw = ImageDraw.Draw(img)
                min_x = CAP_PROP_FRAME_WIDTH
                min_y = CAP_PROP_FRAME_HEIGHT

                for i in range(0, len(xy), 2):
                    x = xy[i:i + 2][0]
                    y = xy[i:i + 2][1]
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    r = 3
                    draw.ellipse((x - r, y - r, x + r, y + r), fill=255)

                crop_img = img.crop((min_x - 200, min_y - 200, min_x + 200, min_y + 200))
                crop_img_after = crop_img.rotate(45)

                fig, axs = plt.subplots(1, 2, figsize=(10, 10))

                # Display the first image in the first subplot
                axs[0].imshow(crop_img, cmap='gray')
                axs[0].set_title('origin')

                # Display the second image in the second subplot
                axs[1].imshow(crop_img_after, cmap='gray')
                axs[1].set_title('noise')
                # plt.imshow(crop_img, cmap='gray')
                # plt.imshow(crop_img_after, cmap='gray')

                # Make a prediction on the new data
                img_transformed = transform(crop_img_after)
                output = load_model(img_transformed.unsqueeze(0))
                _, pred = torch.max(output, 1)
                # Print the predicted class
                print('Predicted label:', pred.item())


plt.show()
