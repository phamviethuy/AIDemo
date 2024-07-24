using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Microsoft.Win32;
using OpenVinoSharp;
using OpenVinoSharp.Extensions.utility;
using System.IO;
using System.Windows;

namespace AIDemo
{



    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const string Model_Path = "D:/Workspace/Study/anomalib-demo/results/Padim/MVTec/bottle/latest/weights/onnx/model.onnx";
        private const string OPEN_VINO_Model_BIN_Path = "D:/Workspace/Study/anomalib-demo/results/Padim/MVTec/bottle/latest/weights/openvino/model.bin";
        private const string OPEN_VINO_Model_Path = "D:/Workspace/Study/anomalib-demo/results/Padim/MVTec/bottle/latest/weights/openvino/model.xml";
        private int currentIndex = 0;
        private List<string> imagePaths = [];
        public MainWindow()
        {
            InitializeComponent();

            OpenVinoSharp.Version version = Ov.get_openvino_version();

            Slog.INFO("---- OpenVINO INFO----");
            Slog.INFO("Description : " + version.description);
            Slog.INFO("Build number: " + version.buildNumber);
            string image_path = @"D:/Workspace/Study/anomalib/datasets/MVTec/bottle/test/broken_small/000.png";
            Predict(image_path);
        }

        private void btnBack_Click(object sender, RoutedEventArgs e)
        {
            var nextImgPath = GetNextImage();
            Predict(nextImgPath);
        }

        private void btnBrower_Click(object sender, RoutedEventArgs e)
        {
            OpenFolderDialog openFolderDialog = new OpenFolderDialog();
            var res = openFolderDialog.ShowDialog();
            if (res != true)
            {
                return;
            }
            tbFolder.Text = openFolderDialog.FolderName;
            imagePaths.Clear();
            imagePaths.AddRange(GetAllImageInFolder(openFolderDialog.FolderName));
            currentIndex = 0;

            Predict(imagePaths[currentIndex]);
        }

        private void btnNext_Click(object sender, RoutedEventArgs e)
        {
            var nextImgPath = GetNextImage();
            Predict(nextImgPath);
        }

        private List<string> GetAllImageInFolder(string folderPath)
        {
            List<string> imagePaths = [];

            // Validate folder path
            if (Directory.Exists(folderPath))
            {
                try
                {
                    // Get image files with common extensions (JPG, JPEG, PNG, GIF)
                    imagePaths.AddRange(
                        Directory.EnumerateFiles(folderPath, "*.{jpg,jpeg,png,gif}", SearchOption.AllDirectories)
                    );
                }
                catch (UnauthorizedAccessException ex)
                {
                    Console.WriteLine("Error: Access denied to folder: {0}", folderPath);
                    Console.WriteLine(ex.Message);
                }
                catch (DirectoryNotFoundException ex)
                {
                    Console.WriteLine("Error: Folder not found: {0}", folderPath);
                    Console.WriteLine(ex.Message);
                }
            }
            else
            {
                Console.WriteLine("Error: Folder does not exist: {0}", folderPath);
            }

            return imagePaths;
        }

        private string GetBackImage()
        {
            if (currentIndex > 0)
            {
                return imagePaths[currentIndex--];
            }
            currentIndex = imagePaths.Count - 1; ;
            return imagePaths[currentIndex];
        }

        private string GetNextImage()
        {
            if (currentIndex < imagePaths.Count)
            {
                return imagePaths[currentIndex++];
            }
            currentIndex = 0;
            return GetNextImage();
        }

        private void Predict(string imagePath)
        {
            using Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);
            System.Drawing.Size inputSize = new System.Drawing.Size(224, 224); // Adjust according to your model's input size
            using Mat resizedImage = new Mat();
            CvInvoke.Resize(image, resizedImage, inputSize);
            Mat blob = DnnInvoke.BlobFromImage(resizedImage, 1.0, inputSize, new MCvScalar(0, 0, 0), swapRB: true, crop: false);

            //// Set the input blob to the network
            //net.SetInput(blob);

            //// Perform inference
            //Mat output = net.Forward();

            using Core core = new Core();
            using OpenVinoSharp.Model model = core.read_model(OPEN_VINO_Model_Path, OPEN_VINO_Model_BIN_Path);
            using CompiledModel compiled_model = core.compile_model(model, "AUTO");

            var input = compiled_model.input(0);
            var output = compiled_model.output(0);
            using InferRequest infer_request = compiled_model.create_infer_request();
            using Tensor input_tensor = infer_request.get_tensor("input");

            infer_request.infer();
            using Tensor output_tensor = infer_request.get_tensor("output0");
        }
   
        public Mat PreProcess(Mat image)
        {
            Mat processedImage = image.Clone();

            if (processedImage.NumberOfChannels == 3)
            {
                // Convert from BGR to RGB
                CvInvoke.CvtColor(processedImage, processedImage, Emgu.CV.CvEnum.ColorConversion.Bgr2Rgb);
            }

            // If image is 3 channels (RGB), transpose dimensions
            if (processedImage.NumberOfChannels == 3)
            {
                // Convert to 4D tensor (Batch, Channel, Height, Width)
                // Since Emgu.CV's Mat class doesn't directly support this, 
                // you might need additional logic depending on your specific needs.

                // For now, we will just return the image with correct channels
                // assuming the Mat is in RGB format.
            }

            return processedImage;
        }

        private void SetImage(Mat image)
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                PreviewImage.Source = image.ToBitmapSource();
            });
        }
    }
}