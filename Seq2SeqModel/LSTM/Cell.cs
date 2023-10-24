using Seq2SeqModel.Entities;
using Seq2SeqModel.Extentions;

namespace Seq2SeqModel.LSTM
{
    [Serializable]
    public class Cell
    {
        //Weight matrix for the input gate, used to control how much of the new input is used to update the cell state.
        public Weights WeightInputGate { get; set; }
        //Weight matrix for the input gate, used to control how much of the previous hidden state is used to update the cell state.
        public Weights WeightInputHidden { get; set; }
        // Bias for the input gate.
        public Weights BiasInput { get; set; }

        //Weight matrix for the forget gate, used to control how much of the previous cell state is forgotten.
        public Weights WeightForgetGate { get; set; }
        //Weight matrix for the forget gate, used to control how much of the previous hidden state is used to forget the previous cell state.
        public Weights WeightForgetHidden { get; set; }
        //Bias for the forget gate.
        public Weights BiasForget { get; set; }

        //Weight matrix for the output gate, used to control how much of the current cell state is output as the hidden state.
        public Weights WeightOutputGate { get; set; }
        //Weight matrix for the output gate, used to control how much of the previous hidden state is used to determine the output.
        public Weights WeightOutputHidden { get; set; }
        //Bias for the output gate.
        public Weights BiasOutput { get; set; }

        //Weight matrix for the cell write operation, used to control how much of the new input is written to the cell state.
        public Weights WeightCellGate { get; set; }
        //Weight matrix for the cell write operation, used to control how much of the previous hidden state is used to write to the cell state.
        public Weights WeightCellHidden { get; set; }
        //Bias for the cell write operation.
        public Weights BiasCell { get; set; }

        public Weights HiddenState { get; set; }
        public Weights CellState { get; set; }


        public int HiddenDim { get; set; }
        public int Dim { get; set; }


        public Cell(int hiddenDim, int dim)
        {

            WeightInputGate = new Weights(dim, hiddenDim, true);
            WeightInputHidden = new Weights(hiddenDim, hiddenDim, true);
            BiasInput = new Weights(1, hiddenDim, 0);

            WeightForgetGate = new Weights(dim, hiddenDim, true);
            WeightForgetHidden = new Weights(hiddenDim, hiddenDim, true);
            BiasForget = new Weights(1, hiddenDim, 0);

            WeightOutputGate = new Weights(dim, hiddenDim, true);
            WeightOutputHidden = new Weights(hiddenDim, hiddenDim, true);
            BiasOutput = new Weights(1, hiddenDim, 0);

            WeightCellGate = new Weights(dim, hiddenDim, true);
            WeightCellHidden = new Weights(hiddenDim, hiddenDim, true);
            BiasCell = new Weights(1, hiddenDim, 0);

            HiddenState = new Weights(1, hiddenDim, 0);
            CellState = new Weights(1, hiddenDim, 0);
            HiddenDim = hiddenDim;
            Dim = dim;
        }

        public Weights Step(Weights input, GraphComputations innerGraph)
        {
            var hidden_prev = HiddenState;
            var cellPrev = CellState;

            var cell = this;
            var weight0 = innerGraph.Multiply(input, cell.WeightInputGate);
            var weight1 = innerGraph.Multiply(hidden_prev, cell.WeightInputHidden);
            var inputGate = innerGraph
                .SigmoidCalculations(innerGraph
                                    .AddWeights(innerGraph.AddWeights(weight0, weight1), cell.BiasInput));

            var weight2 = innerGraph.Multiply(input, cell.WeightForgetGate);
            var weight3 = innerGraph.Multiply(hidden_prev, cell.WeightForgetHidden);
            var forgetGate = innerGraph
                .SigmoidCalculations(innerGraph
                                    .AddWeights(innerGraph.AddWeights(weight2, weight3), cell.BiasForget));

            var weight4 = innerGraph.Multiply(input, cell.WeightOutputGate);
            var weight5 = innerGraph.Multiply(hidden_prev, cell.WeightOutputHidden);
            var outputGate = innerGraph
                .SigmoidCalculations(innerGraph
                                    .AddWeights(innerGraph.AddWeights(weight4, weight5), cell.BiasOutput));

            var weight6 = innerGraph.Multiply(input, cell.WeightCellGate);
            var weight7 = innerGraph.Multiply(hidden_prev, cell.WeightCellHidden);
            var cellWrite = innerGraph
                .Tanh(innerGraph
                    .AddWeights(innerGraph.AddWeights(weight6, weight7), cell.BiasCell));

            // compute new cell activation
            var retainCell = innerGraph.ElementalMultiply(forgetGate, cellPrev); // what do we keep from cell
            var writeCell = innerGraph.ElementalMultiply(inputGate, cellWrite); // what do we write to cell
            var newCell = innerGraph.AddWeights(retainCell, writeCell); // new cell contents


            // compute hidden state as gated, saturated cell activations
            var hiddenState = innerGraph.ElementalMultiply(outputGate, innerGraph.Tanh(newCell));

            HiddenState = hiddenState;
            CellState = newCell;
            return HiddenState;
        }


        public virtual List<Weights> GetData()
        {
            List<Weights> result = new List<Weights>
            {
                BiasCell,
                BiasForget,
                BiasInput,
                BiasOutput,

                WeightCellHidden,
                WeightCellGate,

                WeightForgetHidden,
                WeightForgetGate,

                WeightInputHidden,
                WeightInputGate,

                WeightOutputHidden,
                WeightOutputGate
            };

            return result;
        }

        public void Reset()
        {
            HiddenState = new Weights(1, HiddenDim, 0);
            CellState = new Weights(1, HiddenDim, 0);
        }
    }
}
