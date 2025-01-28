import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage
from torchvision.transforms.functional import to_tensor, resize, to_pil_image


class ImageShadowRemoval(Node):

    def __init__(self):
        super().__init__('image_shadow_removal')

        # Subscription to an image topic
        self.subscription = self.create_subscription(
            Image,
            '/image_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publisher for processed images
        self.publisher = self.create_publisher(Image, '/processed_image', 10)

        # Initialize ONNX model
        onnx_model_path = "output/deshadow_model.onnx"
        self.sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

        # Bridge for ROS <-> OpenCV conversions
        self.bridge = CvBridge()

    def single_image_inference_onnx(self, cv_image):
        # Convert OpenCV image to PIL format
        input_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        original_size = input_image.size  # Store original image size

        # Preprocess input image
        input_tensor = to_tensor(resize(input_image, (256, 256)))  # Resize to model input size
        input_tensor = input_tensor.unsqueeze(0).numpy()  # Add batch dimension and convert to NumPy array

        # Perform inference
        inputs = {self.input_name: input_tensor}
        onnx_output = self.sess.run(None, inputs)

        # Postprocess: Convert output back to PIL image and resize to original size
        result_tensor = np.clip(onnx_output[0], 0, 1)  # Ensure values are within [0, 1]
        result_image = to_pil_image(torch.from_numpy(result_tensor).squeeze(0))  # Convert to PIL image
        result_image = result_image.resize(original_size, PILImage.BILINEAR)  # Resize to original size

        # Convert PIL image to OpenCV format
        result_cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        return result_cv_image

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Run the ONNX model for shadow removal
            processed_image = self.single_image_inference_onnx(cv_image)

            # Convert the processed OpenCV image back to ROS Image message
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')

            # Publish the processed image
            self.publisher.publish(processed_msg)
            self.get_logger().info('Processed image published.')

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')


def main(args=None):
    rclpy.init(args=args)

    image_shadow_removal = ImageShadowRemoval()

    try:
        rclpy.spin(image_shadow_removal)
    except KeyboardInterrupt:
        image_shadow_removal.get_logger().info('Node interrupted by user')
    finally:
        # Destroy the node explicitly
        image_shadow_removal.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()