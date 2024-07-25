using AIDemo.Models;
using OpenCvSharp;
using OpenVinoSharp;
using OpenVinoSharp.Extensions.model;
using OpenVinoSharp.Extensions.process;
using OpenVinoSharp.Extensions.result;
using OpenVinoSharp.Extensions.utility;
using System.IO;

namespace AIDemo.Services
{
    internal class Anomalib8DetConfig : Config
    {
        public int batch_num = 1;
        public string cache_dir = "model/";
        public int categ_nums = 80;
        public long[] input_size = [1L, 3L, 256L, 256L];
        public bool use_gpu;
        public Mat anomaly_map { get; set; }
        public string device { get; set; } = "CPU";
        public string model_bin_path { get; set; } = "";
        public string model_xml_path { get; set; } = "";
        public bool pre_lable { get; set; }
        public double pre_score { get; set; }

        public void set_model(string model_path)
        {
            FileInfo fi = new FileInfo(model_path);
            if (fi.Extension == "xml")
            {
                model_xml_path = model_path;
                model_bin_path = fi.FullName.Replace(".xml", ".bin");

                return;
            }
            if (fi.Extension == "bin")
            {
                model_bin_path = model_path;
                model_xml_path = fi.FullName.Replace(".bin", ".xml");
                return;
            }
        }
    }

    internal class AnomalyPredictor : Predictor
    {
        private long[] m_input_size;

        public AnomalyPredictor(Anomalib8DetConfig config) : base(config.model_bin_path, config.device, config.cache_dir, config.use_gpu, config.input_size)
        {
            m_input_size = config.input_size;
        }

        public DetResult predict(Mat image)
        {
            //Mat mat = new Mat();
            //Cv2.CvtColor(image, mat, ColorConversions.Bgr2Rgb);
            //var m_factors = new float[1];
            //mat = Resize.letterbox_img(mat, (int)m_input_size[2], out m_factors[0]);
            //mat = Normalize.run(mat, is_scale: true);
            //float[] input_data = Permute.run(mat);
            //float[] result = infer(input_data, [1, 3, 900, 900]);

            //todo
            return new DetResult();
        }
    }

    internal class OpenVinoInferencer : Inferencer
    {
        private readonly AnomalyPredictor _predictor;
        private readonly Core core;
        private readonly MetaData metaData;

        public OpenVinoInferencer(string modelPath, MetaData metaData)
        {
            OpenVinoSharp.Version version = Ov.get_openvino_version();
            Slog.INFO("---- OpenVINO INFO----");
            Slog.INFO("Description : " + version.description);
            Slog.INFO("Build number: " + version.buildNumber);

            this.metaData = metaData ?? new MetaData();
            modelPath += "model.xml";
            core = new Core();
            Anomalib8DetConfig config = new();
            config.set_model(modelPath);
            _predictor = new(config);
        }

        public void Predict(string imgPath)
        {
            Mat image = Cv2.ImRead(imgPath);
            DetResult result = _predictor.predict(image);
            Mat result_im = Visualize.draw_det_result(result, image);
            Cv2.ImShow("Result", result_im);
            Cv2.WaitKey(0);
        }
    }
}