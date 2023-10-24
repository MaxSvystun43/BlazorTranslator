using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;
using Seq2SeqModel.LSTM;

namespace Seq2SeqModel.Model
{
    [Serializable]
    public class Encoder
    {
        public List<Cell> Encoders = new List<Cell>();
        //hidden dimension size of the encoder.
        public int HiddenDim { get; set; }
        //input dimension size of the encoder.
        public int Dim { get; set; }
        //depth or number of cells/layers in the encoder.
        public int Depth { get; set; }

        public Encoder(int hdim, int dim, int depth)
        {
            Encoders.Add(new Cell(hdim, dim));

            HiddenDim = hdim;
            Dim = dim;
            Depth = depth;
        }

        /// <summary>
        /// Resets the state of each cell in the encoder. It calls the <c>Reset</c> method of each <c>Cell</c> object in the <c>Encoders</c> list.
        /// </summary>
        public void Reset()
        {
            foreach (var item in Encoders)
            {
                item.Reset();
            }

        }

        /// <summary>
        /// Performs the encoding process using the given <paramref name="weights"/> and a GraphComputations <paramref name="graph"/>. 
        /// It iterates through each <c>Cell</c> in the <с>Encoders</с> list and applies the Step method of the Cell to update the weights. 
        /// The updated <paramref name="weights"/> are passed as input to the next cell in the iteration. The final updated <paramref name="weights"/> are returned.
        /// </summary>
        /// <param name="weights"></param>
        /// <param name="graph"></param>
        /// <returns></returns>
        public Weights Encode(Weights weights, GraphComputations graph)
        {
            foreach (var encoder in Encoders)
            {
                var tempWeight = encoder.Step(weights, graph);
                weights = tempWeight;
            }
            return weights;
        }

        public List<Weights> GetData()
        {
            List<Weights> result = new List<Weights>();

            foreach (var item in Encoders)
            {
                result.AddRange(item.GetData());
            }
            return result;
        }
    }
}
