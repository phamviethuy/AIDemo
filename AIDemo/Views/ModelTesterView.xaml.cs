using System.Windows.Controls;

namespace Views
{
    /// <summary>
    /// Interaction logic for ModelTesterView.xaml
    /// </summary>
    public partial class ModelTesterView : UserControl
    {
        public ModelTesterView()
        {
            InitializeComponent();
            Loaded += ModelTesterView_Loaded;
        }

        private void ModelTesterView_Loaded(object sender, System.Windows.RoutedEventArgs e)
        {
            Focus();
        }
    }
}