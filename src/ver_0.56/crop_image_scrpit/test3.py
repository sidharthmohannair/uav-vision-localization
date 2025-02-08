import cv2
from PIL import Image
import os
import numpy as np

def crop_around_click(image_path, crop_size):
    # Get current directory
    current_dir = os.getcwd()
    click_pos = []
    half_size = crop_size // 2
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse click
            click_pos.append((x, y))
            print(f"Clicked position: x={x}, y={y}")
            cv2.destroyAllWindows()

    # Read and display image for clicking
    img = cv2.imread(image_path)
    cv2.imshow('Click where you want to crop', img)
    cv2.setMouseCallback('Click where you want to crop', mouse_callback)
    cv2.waitKey(0)
    
    if click_pos:
        # Crop using PIL for better handling
        x, y = click_pos[0]
        pil_img = Image.open(image_path)
        
        # Calculate crop boundaries
        left = x - half_size
        top = y - half_size
        right = x + half_size
        bottom = y + half_size
        
        # Handle image boundaries
        width, height = pil_img.size
        
        # Adjust coordinates if they go outside image boundaries
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > width:
            left -= (right - width)
            right = width
        if bottom > height:
            top -= (bottom - height)
            bottom = height
            
        # Crop the image
        cropped = pil_img.crop((left, top, right, bottom))
        
        # If cropped area is smaller than desired size, add padding
        if cropped.size != (crop_size, crop_size):
            new_img = Image.new(pil_img.mode, (crop_size, crop_size), (255, 255, 255))  # White background
            paste_x = (crop_size - cropped.size[0]) // 2
            paste_y = (crop_size - cropped.size[1]) // 2
            new_img.paste(cropped, (paste_x, paste_y))
            cropped = new_img
        
        # Save cropped image in current directory
        output_path = os.path.join(current_dir, f'cropped_{crop_size}px.jpg')
        cropped.save(output_path)
        print(f"Cropped image saved as '{output_path}'")
        
        # Display cropped image dimensions
        print(f"Cropped image dimensions: {cropped.size}")

# Usage
crop_size = 100  # Change this value to get different sized crops
crop_around_click('drone_image2.jpg', crop_size)
