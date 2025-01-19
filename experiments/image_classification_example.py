import os
import sys
import sklearn
import numpy as np
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from PIL import Image
import matplotlib.pyplot as plt

# Load images
def load_images(image_paths):
    images = []
    valid_paths = []
    for path in image_paths:
        try:
            # Open image
            img = Image.open(path)
            
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                print(f"Converting {path} from {img.mode} to RGB")
                img = img.convert('RGB')
            
            # Resize image
            img = img.resize((224, 224))
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Validate dimensions
            if img_array.shape == (224, 224, 3):
                images.append(img_array)
                valid_paths.append(path)
            else:
                print(f"Skipping {path}: Invalid shape {img_array.shape}")
                
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
            
    return images, valid_paths

# Create a simple classifier function (using random forest as an example)
def classifier_fn(images):
    # Replace with your actual model prediction
    return np.random.random((len(images), 2))

# Main execution flow
def main():
    # Set image path
    image_dir = os.path.join('doc', 'images')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Load and preprocess images
    images, valid_paths = load_images(image_paths)
    
    if not images:
        print("No valid images found!")
        return
        
    print(f"Successfully loaded {len(images)} images")
    
    # Create explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Create explanation for each image
    for i, (image, path) in enumerate(zip(images, valid_paths)):
        print(f"\nProcessing image {i+1}/{len(images)}: {path}")
        
        try:
            # Create segmenter
            segmenter = SegmentationAlgorithm('quickshift', 
                                            kernel_size=4, 
                                            max_dist=200, 
                                            ratio=0.2,
                                            convert2lab=True)
            
            # Get explanation
            explanation = explainer.explain_instance(image, 
                                                  classifier_fn,
                                                  top_labels=5,
                                                  hide_color=0,
                                                  num_samples=1000,
                                                  segmentation_fn=segmenter)
            
            # Display original image and explanation
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(image)
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.title('LIME Explanation')
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], 
                positive_only=True, 
                num_features=5, 
                hide_rest=True
            )
            plt.imshow(temp)
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error explaining {path}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 
