using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;
using Seq2SeqModel.Model;

namespace Seq2SeqModel.AttentionModel
{
    [Serializable]
    public class AttentionDecoder
    {
        public List<AttentionDecoderCell> decoders = new List<AttentionDecoderCell>();
        private int hdim { get; set; }
        private int dim { get; set; }
        private int depth { get; set; }
        private Unit Attention { get; set; }

        public AttentionDecoder(int hdim, int dim, int depth)
        {
            decoders.Add(new AttentionDecoderCell(hdim, dim));
            for (int i = 1; i < depth; i++)
            {
                decoders.Add(new AttentionDecoderCell(hdim, hdim));

            }
            Attention = new Unit(hdim);
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }
        public void Reset()
        {
            foreach (var item in decoders)
            {
                item.Reset();
            }
        }

        public Weights Decode(Weights input, List<Weights> encoderOutput, GraphComputations graph)
        {
            var V = input;
            var lastStatus = this.decoders.FirstOrDefault().CellState;
            var context = Attention.Perform(encoderOutput, lastStatus, graph);
            foreach (var encoder in decoders)
            {
                var e = encoder.Step(context, V, graph);
                V = e;
            }

            return V;
        }

        public List<Weights> GetData()
        {
            List<Weights> result = new List<Weights>();

            foreach (var item in decoders)
            {
                result.AddRange(item.GetData());
            }
            result.AddRange(Attention.GetData());
            return result;
        }

    }
}
