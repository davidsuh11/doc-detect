from argparse import ArgumentParser
import torch
import torchvision.transforms as T
import numpy as np
import cv2 as cv
from PIL import Image
from data.augments import preprocess_train
from model import build
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
from rich import print


"""
Predicts segmentation of document in image
Image format must be RGB in range [0, 1]
"""
def predict_image(model, image):
    image = np.array(image).astype(np.float32)/255.
    image = preprocess_train(image=image)['image']
    output = model(image.unsqueeze(0).to(args.device))
    output = output['out'].squeeze(0).cpu().numpy()
    output = np.argmax(output, axis=0)
    return output

def draw_segmentation(image, segmentation):
    color = np.asarray([0, 255, 0])
    masked_image = np.where(segmentation[..., None] == 1, color, image).astype(np.uint8)
    return cv.addWeighted(image, 0.8, masked_image, 0.2, 0.)

def get_segmentation_rectangle(segmentation):
    segmentation = np.uint8(segmentation)
    contours, _ = cv.findContours(segmentation, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0): return None
    return cv.minAreaRect(max(contours, key=cv.contourArea))

"""
Draws the minimum area rectangle bounding the segmentation
"""
def draw_segmentation_rectangle(image, segmentation):
    color = np.asarray([0, 255, 0])
    rect = get_segmentation_rectangle(segmentation)
    if not rect: return image
    
    rect = cv.boxPoints(rect)
    rect = np.int0(rect)
    cv.drawContours(image, [rect], 0, (0,255,0), 2)
    return image

def main(args):
    if not os.path.exists('./out'):
        os.mkdir('./out')

    model = build(pretrained=True)
    model.load_state_dict(torch.load(f'./checkpoints/{args.model}.pth'))
    model = model.to(args.device)
    
    model.eval()
    with torch.no_grad():

        if args.image:
            image = np.array(Image.open(args.image))
            output = predict_image(model, image)
            output = cv.resize(output, (image.shape[1], image.shape[0]), interpolation=cv.INTER_NEAREST)
            image = draw_segmentation(image, output)
            image = draw_segmentation_rectangle(image, output)
            Image.fromarray(image).save(os.path.join(args.save, args.image.split('/')[-1]+ '.result.jpg'))
        elif args.video:
            vid = cv.VideoCapture(args.video)
            frames = []
            while True:
                ret, frame = vid.read()
                if not ret: break
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                output = predict_image(model, frame)
                output = cv.resize(output, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_NEAREST)
                frame = draw_segmentation(frame, output)
                frame = draw_segmentation_rectangle(frame, output)
                frames.append(frame)
            vid.release()
            out = cv.VideoWriter(os.path.join(args.save, args.video.split('/')[-1]+ '.result.mp4'), 
                                cv.VideoWriter_fourcc(*'mp4v'), 15, (frames[0].shape[1], frames[0].shape[0]))
            for frame in frames:
                out.write(frame)
            out.release()


if __name__ == '__main__':
    parser = ArgumentParser(prog = 'Execute inference on a trained model')
    parser.add_argument('--model', type=str, default='model', help='Name of model')
    parser.add_argument('--image', type=str, default=None, help='Path to image')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for prediction')
    parser.add_argument('--video', type=str, default=None, help='Path to video')
    parser.add_argument('--save', type=str, default='out', help='Save output to file')
    args = parser.parse_args()
    main(args)
