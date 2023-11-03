import torch
import torch.nn as nn
import torchvision
import numpy as np

# class PoseEstimationTransformer(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
#         super(PoseEstimationTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, hidden_dim)
#         self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers)
#         self.fc = nn.Linear(hidden_dim, num_keypoints * 2)  # Assuming num_keypoints for pose estimation

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.transformer(x)
#         x = self.fc(x)
#         return x
class SpatialTransformerTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, num_classes):
        super(SpatialTransformerTransformer, self).__init__()

        # Spatial Transformer Module
        self.localization_network = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.localization_fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)  # 2D affine transformation matrix (2x3)
        )

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # Classification Head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Spatial Transformer Module
        localization_params = self.localization_fc(self.localization_network(x).view(x.size(0), -1))
        theta = localization_params.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        # Transformer Encoder
        x = x.permute(2, 0, 1)  # Transformer expects sequence dimension as the first dimension
        x = self.transformer_encoder(x)

        # Classification Head
        x = x.permute(1, 2, 0)  # Reshape for classification
        x = self.fc(x)

        return x
      
def extract_keypoints(image):
    keypoints = pose_estimation_model(image)
    return keypoints

def data_augmentation(image, keypoints):
    augmented_image, augmented_keypoints = perform_augmentation(image, keypoints)
    return augmented_image, augmented_keypoints
  
def normalize_coordinates(keypoints, image_width, image_height):
    keypoints[:, 0] /= image_width
    keypoints[:, 1] /= image_height
    return keypoints
  
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = PoseEstimationTransformer(input_dim, hidden_dim, num_heads, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for batch in data_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Validation loop (if needed)
    model.eval()
    with torch.no_grad():
        validation_loss = 0
        for batch in validation_loader:
            inputs, targets = batch
            outputs = model(inputs)
            validation_loss += criterion(outputs, targets)

    print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item()} - Validation Loss: {validation_loss.item()}')

