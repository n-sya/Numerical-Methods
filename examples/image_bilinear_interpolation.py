import matplotlib.pyplot as plt
import numpy as np


# Downsampling and enlargement factors
downsampling_factor = 5
enlargement_factor = 8


# Load the original image
original_image = plt.imread("data/flower.jpg")

# Convert image to floating point
image = original_image.astype(float)

# Normalise if the image uses values from 0 to 255
if image.max() > 1.0:
    image /= 255.0


# Downsample the original image
shrunk_image = image[
    ::downsampling_factor,
    ::downsampling_factor,
    :3,
]

shrunk_height, shrunk_width, channels = shrunk_image.shape


# Calculate the enlarged image dimensions
enlarged_height = (
    (shrunk_height - 1) * enlargement_factor + 1
)

enlarged_width = (
    (shrunk_width - 1) * enlargement_factor + 1
)

enlarged_image = np.zeros(
    (
        enlarged_height,
        enlarged_width,
        channels,
    ),
    dtype=float,
)


# Apply bilinear interpolation
for i in range(enlarged_height):
    y = i / enlargement_factor

    y1 = int(np.floor(y))
    y2 = min(y1 + 1, shrunk_height - 1)

    dy = y - y1

    for j in range(enlarged_width):
        x = j / enlargement_factor

        x1 = int(np.floor(x))
        x2 = min(x1 + 1, shrunk_width - 1)

        dx = x - x1

        top = (
            (1 - dx) * shrunk_image[y1, x1]
            + dx * shrunk_image[y1, x2]
        )

        bottom = (
            (1 - dx) * shrunk_image[y2, x1]
            + dx * shrunk_image[y2, x2]
        )

        enlarged_image[i, j] = (
            (1 - dy) * top
            + dy * bottom
        )


# Display the results
figure, axes = plt.subplots(
    1,
    3,
    figsize=(12, 5),
)

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(shrunk_image)
axes[1].set_title("Downsampled Image")
axes[1].axis("off")

axes[2].imshow(enlarged_image)
axes[2].set_title("Bilinear Interpolation")
axes[2].axis("off")

plt.tight_layout()
plt.show()