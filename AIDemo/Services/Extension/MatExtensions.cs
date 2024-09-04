using OpenCvSharp;

namespace Services.Extension
{
    // Extension method to permute axes, similar to numpy.transpose
    public static class MatExtensions
    {
        public static Mat PermuteAxes(this Mat mat, params int[] axes)
        {
            var sizes = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                sizes[i] = mat.Size(axes[i]);
            }
            return mat.Reshape(0, sizes);
        }
    }
}