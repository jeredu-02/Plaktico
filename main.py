import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pickle

class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Automatically detect labels from subdirectories
        self.labels = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Collect image paths and their corresponding labels
        self.image_paths = []
        self.image_labels = []
        
        for label in self.labels:
            label_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(label_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.image_labels.append(self.label_to_idx[label])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class MultiLabelClassifier:
    def __init__(self, labels):
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Modify final layer for multi-label classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(labels))
        
        # Store labels and mapping
        self.labels = labels
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def train(self, train_dir, val_dir, epochs=10, learning_rate=0.001):
        # Create datasets
        train_dataset = MultiLabelDataset(train_dir, transform=self.transform)
        val_dataset = MultiLabelDataset(val_dir, transform=self.transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += loss.item()
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Accuracy: {100 * train_correct / len(train_dataset):.2f}%")
            print(f"Val Loss: {val_loss/len(val_loader):.4f}, "
                  f"Val Accuracy: {100 * val_correct / len(val_dataset):.2f}%")
    
    def save_model(self, filepath='classifier_model.pkl'):
        """
        Save the model using pickle
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'model_state': self.model.state_dict(),
            'labels': self.labels,
            'label_to_idx': self.label_to_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='classifier_model.pkl'):
        """
        Load the model from a pickle file
        
        Args:
            filepath (str): Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Reinitialize the model with original architecture
        num_labels = len(model_data['labels'])
        self.model = models.resnet18(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_labels)
        
        # Load saved state
        self.model.load_state_dict(model_data['model_state'])
        self.labels = model_data['labels']
        self.label_to_idx = model_data['label_to_idx']
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        # Move to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {filepath}")
    
    def predict(self, image_path):
        """
        Predict label for a single image
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            tuple: (predicted_label, confidence)
        """
        # Prepare the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_prob, top_class = probabilities.topk(1, dim=1)
            
            predicted_idx = top_class.item()
            predicted_label = self.idx_to_label[predicted_idx]
            confidence = top_prob.item()
        
        return predicted_label, confidence

# Example usage
def main():
    # Path to your image directories
    train_dir = 'archive/dataset/train'  # Directory with 6 label subdirectories
    val_dir = 'archive/dataset/test'  # Directory with 6 label subdirectories
    
    # Automatically detect labels from training directory
    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Initialize classifier
    classifier = MultiLabelClassifier(labels)
    
    # Train the model
    classifier.train(train_dir, val_dir, epochs=15)
    
    # Save the model
    classifier.save_model('multi_label_classifier_2.pkl')
    
    # Optional: Load the model
    # classifier.load_model('multi_label_classifier.pkl')
    
    # Predict on a single image
    # test_image = '/path/to/test/image.jpg'
    # predicted_label, confidence = classifier.predict(test_image)
    # print(f"Predicted Label: {predicted_label}")
    # print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()