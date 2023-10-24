using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;

namespace Seq2SeqModel.AttentionModel
{
    [Serializable]
    public class Unit
    {
        //used to project the intermediate attention scores onto a single scalar value for each input state
        public Weights FinalScore { get; set; }
        public Weights Attention { get; set; }
        public Weights BiasAttention { get; set; }
        public Weights AttentionState { get; set; }
        public Weights BiasAttentionState { get; set; }
        //Stores the index of the maximum attention score after performing the attention mechanism.
        public int MaxIndex { get; set; }
        //Represents the batch size used in the attention computation.
        public int batchSize { get; set; }

        public Unit(int size)
        {
            batchSize = 1;

            Attention = new Weights((size * 2), size, true);

            AttentionState = new Weights(size, size, true);

            BiasAttention = new Weights(1, size, 0);
            BiasAttentionState = new Weights(1, size, 0);

            FinalScore = new Weights(size, 1, true);
        }

        /// <summary>
        /// Performs the attention mechanism using the given <paramref name="input"/>, <paramref name="state"/>, and a <paramref name="graph"/>. 
        /// It iterates through each weight in the input list and calculates intermediate attention scores. 
        /// The attention scores are obtained by applying transformations (multiplication, addition, tanh) on the weights and biases. 
        /// The scores are then multiplied with the FinalScore weights. 
        /// Softmax is applied to normalize the scores, and the resulting weights are stored in <c>res</c>. 
        /// The maximum attention score and its corresponding index (MaxIndex) are determined. 
        /// Finally, the context is calculated by combining the input weights with the normalized attention scores, and it is returned.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="state"></param>
        /// <param name="graph"></param>
        /// <returns></returns>
        public Weights Perform(List<Weights> input, Weights state, GraphComputations graph)
        {
            Weights context;
            List<Weights> attention = new List<Weights>();
            foreach (var weight in input)
            {
                var attWeigth = graph.AddWeights(graph.Multiply(weight, Attention), BiasAttention);  //attention weight
                var attStateWeight = graph.AddWeights(graph.Multiply(state, AttentionState), BiasAttentionState);//attention state weight
                var intermediateScore = graph.Tanh(graph.AddWeights(attWeigth, attStateWeight));
                var attentionRes = graph.Multiply(intermediateScore, FinalScore);
                attention.Add(attentionRes);
            }
            var result = graph.Softmax(attention);

            var maxScore = result[0].Weight[0];
            int maxAttentionScore = 0;
            for (int i = 1; i < result.Count; i++)
            {
                if (result[i].Weight[0] > maxScore)
                {
                    maxScore = result[i].Weight[0];
                    maxAttentionScore = i;
                }
            }
            MaxIndex = maxAttentionScore;

            context = graph.ScaleMultiply(input[0], result[0]);
            for (int i = 1; i < input.Count; i++)
            {
                context = graph.AddWeights(context, graph.ScaleMultiply(input[i], result[i]));
            }
            return context;
        }

        public virtual List<Weights> GetData()
        {
            List<Weights> response = new List<Weights>
            {
                Attention,
                AttentionState,
                BiasAttention,
                BiasAttentionState,
                FinalScore
            };
            return response;
        }
    }
}
