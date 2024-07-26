﻿using AIDemo;
using AIDemo.Models;
using AIDemo.Services;
using Microsoft.Win32;
using Newtonsoft.Json;
using OpenCvSharp.Extensions;
using OpenCvSharp.WpfExtensions;
using Prism.Commands;
using Prism.Mvvm;
using System.IO;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace ViewModels
{
    internal class ModelTesterViewModel : BindableBase
    {
        private readonly List<string> imagePaths = [];
        private string currentImagePath = "";
        private int currentIndex = 0;
        private string imageFolder = "";
        private Inferencer? inferencer;
        private string modelFolder = "";
        private List<ImageSource> outputImages = [];
        private string result = "";
        private double score;

        public ModelTesterViewModel()
        {
            ModelFolder = Settings.Default.ModelFolderPath;
            ImageFolderPath = Settings.Default.ImageFolderPath;
            Init();
            ChooseModelFolderCommand = new DelegateCommand(ChooseModelFolder);
            ChooseImageFolderCommand = new DelegateCommand(ChooseImageFolder);
            NextImageCommand = new DelegateCommand(NextImage);
            BackCommand = new DelegateCommand(BackImage);
        }

        public ICommand BackCommand { get; set; }
        public ICommand ChooseImageFolderCommand { get; set; }
        public ICommand ChooseModelFolderCommand { get; set; }

        public string CurrentImagePath
        { get => currentImagePath; set { SetProperty(ref currentImagePath, value); } }

        public string ImageFolderPath
        { get => imageFolder; set { SetProperty(ref imageFolder, value); } }

        public string ModelFolder
        { get => modelFolder; set { SetProperty(ref modelFolder, value); } }

        public ICommand NextImageCommand { get; set; }

        public List<ImageSource> OutputImages
        { get => outputImages; set { SetProperty(ref outputImages, value); } }

        public string Result
        { get => result; set { SetProperty(ref result, value); } }

        public double Score
        { get => score; set { SetProperty(ref score, value); } }

        private void BackImage()
        {
            var imgPath = GetBackImage();
            Predict(imgPath);
        }

        private string ChooseFolder()
        {
            OpenFolderDialog openFolderDialog = new OpenFolderDialog();
            var res = openFolderDialog.ShowDialog();
            if (res != true)
            {
                return string.Empty;
            }

            return openFolderDialog.FolderName;
        }

        private void ChooseImageFolder()
        {
            var folderPath = ChooseFolder();
            if (string.IsNullOrEmpty(folderPath))
            {
                return;
            }
            ImageFolderPath = folderPath;
            Settings.Default.ImageFolderPath = folderPath;
            Settings.Default.Save();

            InitFolder(ImageFolderPath);
        }

        private void ChooseModelFolder()
        {
            var folderPath = ChooseFolder();
            if (string.IsNullOrEmpty(folderPath))
            {
                return;
            }
            Settings.Default.ModelFolderPath = folderPath;
            Settings.Default.Save();

            Init();
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

        private void Init()
        {
            try
            {
                if (!string.IsNullOrEmpty(Settings.Default.ModelFolderPath))
                {
                    var metaData = JsonConvert.DeserializeObject<MetaData>(File.ReadAllText(Path.Combine(Settings.Default.ModelFolderPath, "metadata.json")));
                    inferencer = new OnnxInferencer(Settings.Default.ModelFolderPath, metaData);
                }
                InitFolder(ImageFolderPath);
            }
            catch (Exception e)
            {
                MessageBox.Show(e.ToString());
            }
        }

        private void InitFolder(string folder)
        {
            if (Directory.Exists(folder))
            {
                imagePaths.Clear();
                imagePaths.AddRange(GetAllImageInFolder(folder));
                currentIndex = 0;
                Predict(imagePaths[currentIndex]);
            }
        }

        private void NextImage()
        {
            var imgPath = GetNextImage();
            Predict(imgPath);
        }

        private void Predict(string imagePath)
        {
            if (inferencer == null)
            {
                return;
            }
            CurrentImagePath = imagePath;
            var res = inferencer.Predict(imagePath);
            Result = $"{res.Label}";
            Score = res.Score;

            OutputImages = [res.Image.ToBitmap().ToBitmapSource(), res.HeatMap.ToBitmap().ToBitmapSource(), res.Mask.ToBitmap().ToBitmapSource(), res.Segmentation.ToBitmap().ToBitmapSource()];
        }
    }
}