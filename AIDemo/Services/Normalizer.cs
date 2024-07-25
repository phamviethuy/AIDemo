using NumSharp;
using TorchSharp;

namespace AIDemo.Services
{
    public class Normalizer
    {
        public static NDArray Normalize(
            NDArray targets,
            double threshold,
            double minVal,
            double maxVal)
        {
            var normalized = ((targets - threshold) / (maxVal - minVal)) + 0.5;

            normalized = np.minimum(normalized, 1);
            normalized = np.maximum(normalized, 0);

            return normalized;
        }

        public static torch.Tensor Normalize(
            torch.Tensor targets,
            double threshold,
            double minVal,
            double maxVal)
        {
            var normalized = ((targets - threshold) / (maxVal - minVal)) + 0.5;

            normalized = torch.minimum(normalized, torch.tensor(1));
            normalized = torch.maximum(normalized, torch.tensor(0));

            return normalized;
        }
    }
}