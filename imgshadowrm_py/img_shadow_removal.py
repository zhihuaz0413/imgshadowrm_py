import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ImageShadowRemoval(Node):

    def __init__(self):
        super().__init__('image_shadow_removal')

        self.subscription = self.create_subscription(
            Image,
            '/image_topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.bridge = CvBridge()

    def listener_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Example processing: Display the image
            self.get_logger().info('Image received, displaying...')
            cv2.imshow("Received Image", cv_image)

            # Wait for a key press for a short duration to display the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info('Closing image window')
                cv2.destroyAllWindows()

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
