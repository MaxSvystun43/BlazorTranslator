using Seq2SeqModel.AttentionModel;
using Seq2SeqModel.Model;

namespace Seq2SeqModel.Entities
{
    [Serializable]
    public class ModelData
    {
        public int MaxSentenceGen = 100;
        public int HiddenSizes;
        public int LetterSize;

        #region optimization

        //the strength  of L2 regularization used in the optimization process.
        public double RegularizationStrength = 0.000001;
        //learning rate used in the optimization process.
        public double LearningRate = 0.01;
        //value at which gradients are clipped during the optimization process.
        public double Clipval = 5.0;

        #endregion

        //weights used in the model.
        public Weights WeightsIn;
        public Encoder Encoder;
        public Encoder ReversEncoder;
        public AttentionDecoder Decoder;
        //Indicates whether dropout is used in the model
        public bool UseDropout { get; set; }

        public Dictionary<string, int> wordToIndex = new Dictionary<string, int>();
        public Dictionary<int, string> indexToWord = new Dictionary<int, string>();


        #region Output Layer Weights
        //the weights used in the output layer of the model.
        public Weights WeightsOut { get; set; }
        //the biases used in the output layer of the model.
        public Weights BiasOut { get; set; }

        #endregion

        //the depth or number of layers in the model.
        public int Depth { get; set; }
    }
}
