import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# # =============================
# # Tensor Basics and Autograd
# # =============================
# x = torch.tensor([1.0, 2.0, 3.0])
# y = torch.ones(3)
# z = x + y
# print(z)

# print(x.shape, x.size(), x.dtype)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = x.to(device)

# x = torch.tensor(3.0, requires_grad=True)
# y = x ** 2
# y.backward()
# print(x.grad)

# # =============================
# # Define a Simple Model
# # =============================
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.linear = nn.Linear(2, 1)  # Changed input features to 2

#     def forward(self, x):
#         return self.linear(x)

# model = SimpleModel()
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# # =============================
# # Training Loop
# # =============================
# for epoch in range(10):
#     optimizer.zero_grad()
#     input_tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
#     target_tensor = torch.tensor([[1.0]], dtype=torch.float32)
#     output = model(input_tensor)
#     loss = criterion(output, target_tensor)
#     loss.backward()
#     optimizer.step()
#     print(f'Epoch {epoch}, Loss: {loss.item()}')

# # =============================
# # DataLoader Example
# # =============================
# data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# targets = torch.tensor([[1.0], [0.0]])
# dataset = TensorDataset(data, targets)
# loader = DataLoader(dataset, batch_size=2, shuffle=True)

# for batch in loader:
#     print(batch)

# # =============================
# # TorchScript Model Saving
# # =============================
# scripted_model = torch.jit.script(model)
# print(scripted_model)
# scripted_model.save("scripted_model.pt")
# loaded_model = torch.jit.load("scripted_model.pt")
# print(loaded_model)

# # =============================
# # Matplotlib Plot Examples
# # =============================
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 25, 30, 40]
# plt.plot(x, y)
# plt.title("Basic Line Plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.show(block=False)  # Non-blocking display
# plt.pause(0.1)  # Small delay to ensure rendering

# categories = ['A', 'B', 'C', 'D']
# values = [3, 7, 5, 10]
# plt.figure()  # Create a new figure to avoid overlap
# plt.bar(categories, values, color='blue')
# plt.title("Bar Chart Example")
# plt.show(block=False)
# plt.pause(0.1)

# x = [5, 7, 8, 10, 15]
# y = [20, 30, 25, 40, 50]
# plt.figure()
# plt.scatter(x, y, color='green', marker='x')
# plt.title("Scatter Plot Example")
# plt.show(block=False)
# plt.pause(0.1)

# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title("Figures and Axes")
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# plt.show(block=False)
# plt.pause(0.1)

# plt.figure()
# plt.plot(x, y, color='red', linestyle='--', marker='o', linewidth=2)
# plt.title("Customized Line Plot")
# plt.show(block=False)
# plt.pause(0.1)

# =============================
# Stable Diffusion Text-to-Image
# =============================
# Hugging Face Token (Replace with your actual token)
authorization_token = "your token here"  # Your token

# Model loading
model_id = "CompVis/stable-diffusion-v1-4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
revision = "fp16" if device == "cuda" else "main"
dtype = torch.float16 if device == "cuda" else torch.float32

try:
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        use_auth_token=authorization_token
    )
    pipe.to(device)
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    exit(1)

# Prompt and generate image
try:
    text_prompt = input("Enter your text prompt (or press Enter for default): ").strip()
    if not text_prompt:
        text_prompt = "A serene mountain landscape at sunset"
    print(f"Using prompt: {text_prompt}")

    # Generate image with error handling
    with torch.no_grad():  # Simplified to always use no_grad for consistency
        print("Generating image...")
        image = pipe(text_prompt, guidance_scale=8.5).images[0]

    # Show result
    plt.figure()
    plt.imshow(image)
    plt.axis('off')
    plt.title("Generated Image")
    plt.show(block=False)
    plt.pause(0.1)
except Exception as e:
    print(f"Error generating image: {e}")
