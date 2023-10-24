using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;
using Seq2SeqModel.LSTM;

namespace Seq2SeqModel.Model
{
    [Serializable]
    public class AttentionDecoderCell : Cell
    {
        //Weight matrix that maps the context vector to the input gate of the LSTM cell.
        public Weights WeightInput { get; set; }
        //Weight matrix that maps the context vector to the forget gate of the LSTM cell.
        public Weights WeightForget { get; set; }
        //Weight matrix that maps the context vector to the output gate of the LSTM cell.
        public Weights WeightOutput { get; set; }
        //Weight matrix that maps the context vector to the cell write operation in the LSTM cell.
        public Weights WeightCell { get; set; }


        public AttentionDecoderCell(int hiddenDim, int dim)
            : base(hiddenDim, dim)
        {
            int contextSize = hiddenDim * 2;
            WeightInput = new Weights(contextSize, hiddenDim, true);
            WeightForget = new Weights(contextSize, hiddenDim, true);
            WeightOutput = new Weights(contextSize, hiddenDim, true);
            WeightCell = new Weights(contextSize, hiddenDim, true);

        }

        public Weights Step(Weights context, Weights input, GraphComputations innerGraph)
        {

            var hidden_prev = HiddenState;
            var prevCellState = CellState;

            var cell = this;
            var weight0 = innerGraph.Multiply(input, cell.WeightInputGate);
            var wegith1 = innerGraph.Multiply(hidden_prev, cell.WeightInputHidden);
            var attention01 = innerGraph.Multiply(context, cell.WeightInput);
            var inputGate = innerGraph
                .SigmoidCalculations(innerGraph
                .AddWeights(innerGraph.AddWeights(innerGraph.AddWeights(weight0, wegith1), attention01), cell.BiasInput));


            var weight2 = innerGraph.Multiply(input, cell.WeightForgetGate);
            var weight3 = innerGraph.Multiply(hidden_prev, cell.WeightForgetHidden);
            var attention23 = innerGraph.Multiply(context, cell.WeightForget);
            var forgetGate = innerGraph
                .SigmoidCalculations(innerGraph
                .AddWeights(innerGraph.AddWeights(innerGraph.AddWeights(weight3, weight2), attention23), cell.BiasForget));


            var weight4 = innerGraph.Multiply(input, cell.WeightOutputGate);
            var weight5 = innerGraph.Multiply(hidden_prev, cell.WeightOutputHidden);
            var attention45 = innerGraph.Multiply(context, cell.WeightOutput);
            var outputGate = innerGraph
                .SigmoidCalculations(innerGraph
                .AddWeights(innerGraph.AddWeights(innerGraph.AddWeights(weight5, weight4), attention45), cell.BiasOutput));


            var weight6 = innerGraph.Multiply(input, cell.WeightCellGate);
            var weight7 = innerGraph.Multiply(hidden_prev, cell.WeightCellHidden);
            var attention67 = innerGraph.Multiply(context, cell.WeightCell);
            var cellWrite = innerGraph
                .Tanh(innerGraph
                .AddWeights(innerGraph.AddWeights(innerGraph.AddWeights(weight7, weight6), attention67), cell.BiasCell));


            var retainCell = innerGraph.ElementalMultiply(forgetGate, prevCellState); // what do we keep from cell
            var writeCell = innerGraph.ElementalMultiply(inputGate, cellWrite); // what do we write to cell
            var newCell = innerGraph.AddWeights(retainCell, writeCell); // new cell contents

            var hiddenState = innerGraph.ElementalMultiply(outputGate, innerGraph.Tanh(newCell));

            HiddenState = hiddenState;
            CellState = newCell;

            return HiddenState;
        }

        public override List<Weights> GetData()
        {
            List<Weights> result = new List<Weights>();
            result.AddRange(base.GetData());

            result.Add(WeightInput);
            result.Add(WeightForget);
            result.Add(WeightOutput);
            result.Add(WeightCell);
            return result;
        }
    }
}
