from dataset import *

## creating datasets
train_dataset = FlowerDataset(data_dir,train_ids,labels,train_transform)
val_dataset = FlowerDataset(data_dir,val_ids,labels,val_transform)
test_dataset = FlowerDataset(data_dir,test_ids,labels,val_transform)

## creating dataloaders
train_loader = DataLoader(train_dataset,batch_size = 32,shuffle = True, num_workers = 4)
val_loader = DataLoader(val_dataset,batch_size = 32,shuffle = False,num_workers = 4)
test_loader = DataLoader(test_dataset,batch_size = 32,shuffle = False,num_workers = 4)

from torchvision.models import resnet50, ResNet50_Weights
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,102)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 1e-4)


def train_model(model,criterion,optimizer,num_epochs = 25):
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        
        for inputs,labels in tqdm(train_loader, desc = f'Epoch {epoch+1}/{num_epochs}'):
            inputs,labels = inputs.to(device),labels.to(device)
            labels = labels.long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)
        
        epoch_loss = running_loss/len(train_dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs,labels in val_loader:
                inputs,labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
    return model


"""
TRAINING AND EVALUATING MODEL
"""
def run():
    trained_model = train_model(model,criterion,optimizer)
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')

    # Saving the model
    torch.save(model.state_dict(), 'flower_resnet50.pth')
    
if __name__ == "__main__":
    run()
            