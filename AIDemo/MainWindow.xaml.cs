using AIDemo.Models;
using AIDemo.Services;
using Microsoft.Win32;
using Newtonsoft.Json;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using System.IO;
using System.Windows;
using System.Windows.Media;

namespace AIDemo
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        private const string IMAGE_PATH = "D:/Workspace/Study/anomalib/datasets/MVTec/bottle/test/broken_large/000.png";
        private const string ONNX_MODEL_PATH = "D:/Workspace/Study/anomalib-demo/results/Patchcore/MVTec/bottle/latest/weights/onnx";
        private const string OPEN_VINO_MODEL_PATH = "D:/Workspace/Study/anomalib-demo/results/Patchcore/MVTec/bottle/latest/weights/openvino";
        private const string TORCH_MODEL_PATH = "D:/Workspace/Study/anomalib-demo/results/Patchcore/MVTec/bottle/latest/weights/torch";

        private readonly List<string> imagePaths = [];
        private readonly Inferencer inferencer;
        private int currentIndex = 0;

        public MainWindow()
        {
            InitializeComponent();

            var metaData = JsonConvert.DeserializeObject<MetaData>(File.ReadAllText(Path.Combine(ONNX_MODEL_PATH, "metadata.json")));
            inferencer = new OnnxInferencer(ONNX_MODEL_PATH, metaData);
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
                    imagePaths.AddRange(Directory.EnumerateFiles(folderPath));
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
                return imagePaths[--currentIndex];
            }
            currentIndex = imagePaths.Count - 1; ;
            return imagePaths[currentIndex];
        }

        private string GetNextImage()
        {
            if (currentIndex < imagePaths.Count - 1)
            {
                return imagePaths[++currentIndex];
            }
            currentIndex = 0;
            return GetNextImage();
        }

        private void Predict(string imagePath)
        {
            var res = inferencer.Predict(imagePath);
            if (res.IsNormal)
            {
                btnResult.Content = $"{res.Label} {res.Score}";
                btnResult.Background = new SolidColorBrush(Colors.Green);
            }
            else
            {
                btnResult.Content = $"{res.Label} {res.Score}";
                btnResult.Background = new SolidColorBrush(Colors.Red);
            }
            SetImage(res.Image, res.Mask);
        }



        private void SetImage(Mat image, Mat heatMap)
        {
            Application.Current.Dispatcher.Invoke(() =>
            {
                PreviewImage.Source = image.ToBitmap().ToBitmapSource();
                SegmentImage.Source = heatMap.ToBitmap().ToBitmapSource();
            });
        }
    }
}