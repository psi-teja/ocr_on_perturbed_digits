from PIL import Image
import matplotlib.pyplot as plt
from prepare_dataset import *

tf_model = keras.models.load_model("tf_model")


# image preprocessing before feeding into OCR model
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    img = Image.fromarray(img)
    new_img = img.resize((28, 28))
    X = np.array(new_img.convert('L'))
    X = X.reshape(1, 28, 28, 1)
    return X


digits = list(range(0, 10))
correct_pred = np.zeros(10)

folder = "digits_noised"

for image_path in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, image_path))
    X = preprocess(img)

    # models gives an array with probabilities
    digit = tf_model.predict(X)

    print(f"{image_path} ------ {digit.argmax()}")

    # ground truth
    gt = int(image_path[-5])

    # predicted value
    pred_digit = digit.argmax()

    if gt == pred_digit:
        correct_pred[gt] += 1

# calculating accuracy from number of correct predictions
accuracy = []
for i in range(10):
    accuracy.append((correct_pred[i] / n) * 100)

print("digits:", digits)
print("ocr_accuracy:", accuracy)

# Create the bar plot
plt.bar(digits, accuracy)

# Add labels and title
plt.xlabel('Digits')
plt.ylabel('OCR Accuracy')
plt.title('OCR Accuracy on perturbed digits')

# Label all the points on the x-axis
plt.xticks(digits)

# Save the plot
plt.savefig("plot.png")

# Show the plot
plt.show()
