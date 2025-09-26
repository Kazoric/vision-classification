import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Model(ABC):
    def __init__(self, device=None, lr=0.001, save=False):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.save = save

        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []

        self.best_val_loss = float('inf')
        self.checkpoint_path = "./checkpoints/"
        self.start_epoch = 0

    @abstractmethod
    def build_model(self):
        """À implémenter dans les sous-classes, retourne un nn.Module."""
        pass

    def train(self, train_loader, val_loader=None, epochs=10):
        # with tqdm(total=epochs, desc='📊 Global Progress', position=0) as global_pbar:
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in tqdm(train_loader, desc=f"🎯 Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            acc = correct / total * 100
            print(f"Train Loss: {running_loss:.4f} | Train Acc: {acc:.2f}%")

            self.train_loss.append(running_loss)
            self.train_acc.append(acc)

            if val_loader:
                val_loss = self.evaluate(val_loader)

                if self.save == True and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss)

            print()

                # global_pbar.set_postfix({
                #     'val_loss': self.valid_loss[-1],
                #     'val_acc': self.valid_acc[-1]
                # })
                # global_pbar.update(1)

    def evaluate(self, data_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        print(f"Validation Loss: {running_loss:.4f} | Validation Accuracy: {acc:.2f}%")

        self.valid_loss.append(running_loss)
        self.valid_acc.append(acc)

        return running_loss

    def predict(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
        return predicted

    def plot_acc_loss(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5))

        ax1.plot(self.train_acc)
        ax1.plot(self.valid_acc)
        ax1.set_title('model accuracy')
        ax1.set_xlabel('epochs')
        ax1.legend(['train accuracy', 'val accuracy'])

        ax2.plot(self.train_loss)
        ax2.plot(self.valid_loss)
        ax2.set_title('model loss')
        ax2.set_xlabel('epochs')
        ax2.legend(['train loss', 'val loss'])

        plt.show()

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }
        model_name = f"{self.name}_epoch-{epoch}_loss-{val_loss}.pt"
        path = os.path.join(self.checkpoint_path, model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint saved at epoch {epoch} with val_loss: {val_loss:.4f}")

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_loss = checkpoint['val_loss']
            print(f"📦 Loaded checkpoint from epoch {self.start_epoch} with val_loss: {self.best_val_loss:.4f}")
        else:
            print("ℹ️ No checkpoint found. Starting from scratch.")
