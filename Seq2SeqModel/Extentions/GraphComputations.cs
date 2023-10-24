using Seq2SeqModel.Entities;

namespace Seq2SeqModel.Extentions
{
    public class GraphComputations
    {
        private List<Action> backPropagation = new List<Action>();

        public bool MakeBackprop { get; set; }

        public GraphComputations(bool needBack = true)
        {
            MakeBackprop = needBack;
        }

        public Weights Tanh(Weights weight)
        {
            var result = new Weights(weight.Rows, weight.Columns, 0);
            var weightLength = weight.Weight.Length;
            for (var i = 0; i < weightLength; i++)
            {
                result.Weight[i] = Math.Tanh(weight.Weight[i]);
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < weightLength; i++)
                    {
                        var mwi = result.Weight[i];
                        weight.Gradient[i] += (1.0 - mwi * mwi) * result.Gradient[i];
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights AddColumns(Weights first, Weights second)
        {
            int sx = 1;
            int sy = 1;

            sy = first.Columns + second.Columns;
            sx = first.Rows;

            var result = new Weights(sx, sy, 0);
            var weightLength = first.Weight.Length;
            for (var i = 0; i < first.Rows; i++)
            {
                for (int j = 0; j < first.Columns; j++)
                {
                    var el = first.GetWeight(i, j);
                    result.SetWeight(i, j, el);
                }
            }
            for (var i = 0; i < second.Rows; i++)
            {

                for (int j = first.Columns; j < second.Columns + first.Columns; j++)
                {
                    var el = second.GetWeight(i, j - first.Columns);
                    result.SetWeight(i, j, el);
                }
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < first.Rows; i++)
                    {
                        for (int j = 0; j < first.Columns; j++)
                        {
                            var el = result.GetGradient(i, j);
                            first.AddGradient(i, j, el);
                        }
                    }
                    for (var i = 0; i < second.Rows; i++)
                    {

                        for (int j = first.Columns; j < second.Columns + first.Columns; j++)
                        {
                            var el = result.GetGradient(i, j);
                            second.AddGradient(i, j - first.Columns, el);
                        }
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights GetByRow(Weights weight, int index)
        {
            var numberOfColumns = weight.Columns;
            var result = new Weights(1, numberOfColumns, 0);
            for (int i = 0, n = numberOfColumns; i < n; i++)
            {
                result.Weight[i] = weight.Weight[numberOfColumns * index + i];
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    for (int i = 0, n = numberOfColumns; i < n; i++)
                    {
                        weight.Gradient[numberOfColumns * index + i] += result.Gradient[i];
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1 + Math.Exp(-x));
        }

        public Weights SigmoidCalculations(Weights weight)
        {
            Weights result = new Weights(weight.Rows, weight.Columns, 0);
            var weightLength = weight.Weight.Length;
            for (var i = 0; i < weightLength; i++)
            {
                result.Weight[i] = Sigmoid(weight.Weight[i]);
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < weightLength; i++)
                    {
                        var mwi = result.Weight[i];
                        weight.Gradient[i] += mwi * (1.0 - mwi) * result.Gradient[i];
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }


        public Weights Dropout(Weights weight, double dropProb)
        {
            Random rand = new Random();
            var result = new Weights(weight.Rows, weight.Columns, 0);
            var weightLength = weight.Weight.Length;
            bool[] dropped = new bool[weight.Rows * weight.Columns];
            var newWeight = weight.CloneData();

            for (var i = 0; i < weightLength; i++)
            {
                if (rand.NextDouble() < dropProb)
                {
                    newWeight.Weight[i] = 0;
                    dropped[i] = true;
                }
                else
                {
                    dropped[i] = false;
                }
            }

            result = newWeight;


            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    var grad = result;
                    weight.Gradient = new double[weightLength]; // zero out gradient  data
                    for (var i = 0; i < weightLength; i++)
                    {
                        if (!dropped[i])
                        {
                            weight.Gradient[i] += grad.Gradient[i]; // copy over the gradient
                        }
                    }

                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights Multiply(Weights first, Weights second)
        {
            var firstRows = first.Rows;
            var secondColums = second.Columns;
            var result = new Weights(firstRows, secondColums, 0);
            for (var i = 0; i < first.Rows; i++)
            { // loop over rows of first
                for (var j = 0; j < second.Columns; j++)
                { // loop over cols of second
                    var dot = 0.0;
                    for (var k = 0; k < first.Columns; k++)
                    { // dot product loop
                        dot += first.Weight[first.Columns * i + k] * second.Weight[second.Columns * k + j];
                    }
                    result.Weight[secondColums * i + j] = dot;
                }
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < first.Rows; i++)
                    { // loop over rows of first
                        for (var j = 0; j < second.Columns; j++)
                        { // loop over cols of second
                            for (var k = 0; k < first.Columns; k++)
                            { // dot product loop
                                var b = result.Gradient[secondColums * i + j];
                                first.Gradient[first.Columns * i + k] += second.Weight[second.Columns * k + j] * b;
                                second.Gradient[second.Columns * k + j] += first.Weight[first.Columns * i + k] * b;
                            }
                        }
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights AddWeights(Weights first, Weights second)
        {

            var result = new Weights(first.Rows, first.Columns, 0);
            for (int i = 0, n = first.Weight.Length; i < n; i++)
            {
                result.Weight[i] = first.Weight[i] + second.Weight[i];
            }
            if (MakeBackprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = first.Weight.Length; i < n; i++)
                    {
                        first.Gradient[i] += result.Gradient[i];
                        second.Gradient[i] += result.Gradient[i];
                    }
                };
                backPropagation.Add(backward);
            }
            return result;

        }

        public Weights ElementalMultiply(Weights first, Weights second)
        {

            var result = new Weights(first.Rows, first.Columns, 0);
            for (int i = 0, n = first.Weight.Length; i < n; i++)
            {
                result.Weight[i] = first.Weight[i] * second.Weight[i];
            }
            if (MakeBackprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = first.Weight.Length; i < n; i++)
                    {
                        first.Gradient[i] += second.Weight[i] * result.Gradient[i];
                        second.Gradient[i] += first.Weight[i] * result.Gradient[i];
                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights ScaleMultiply(Weights first, Weights second)
        {

            var result = new Weights(first.Rows, first.Columns, 0);
            for (int i = 0, n = first.Weight.Length; i < n; i++)
            {
                result.Weight[i] = first.Weight[i] * second.Weight[0];
            }
            if (MakeBackprop)
            {

                Action backward = () =>
                {
                    for (int i = 0, n = first.Weight.Length; i < n; i++)
                    {
                        first.Gradient[i] += second.Weight[0] * result.Gradient[i];
                        second.Gradient[0] += first.Weight[i] * result.Gradient[i];

                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public Weights SoftmaxWithCrossEntropy(Weights weight)
        {
            var result = new Weights(weight.Rows, weight.Columns, 0); // probability volume
            var maxValue = double.MinValue;
            for (int i = 0, n = weight.Weight.Length; i < n; i++)
            {
                if (weight.Weight[i] > maxValue)
                    maxValue = weight.Weight[i];
            }

            var sum = 0.0;
            for (int i = 0, n = weight.Weight.Length; i < n; i++)
            {
                result.Weight[i] = Math.Exp(weight.Weight[i] - maxValue);
                sum += result.Weight[i];
            }
            for (int i = 0, n = weight.Weight.Length; i < n; i++)
            {
                result.Weight[i] /= sum;
            }

            return result;
        }

        public List<Weights> Softmax(List<Weights> weight)
        {
            var result = new List<Weights>(); // probability volume

            var maxValue = double.MinValue;
            for (int i = 0, n = weight.Count; i < n; i++)
            {
                if (weight[i].Weight[0] > maxValue)
                    maxValue = weight[i].Weight[0];
                result.Add(new Weights(weight[i].Rows, weight[i].Columns, 0));
            }

            var sum = 0.0;
            for (int i = 0, n = weight.Count; i < n; i++)
            {
                result[i].Weight[0] = Math.Exp(weight[i].Weight[0] - maxValue);
                sum += result[i].Weight[0];
            }
            for (int i = 0, n = weight.Count; i < n; i++)
            {
                result[i].Weight[0] /= sum;
            }

            if (MakeBackprop)
            {
                Action backward = () =>
                {
                    double ss = 0.0;
                    for (int i = 0; i < weight.Count; i++)
                    {
                        weight[i].Gradient[0] += result[i].Gradient[0] * result[i].Weight[0];

                        ss += result[i].Gradient[0] * result[i].Weight[0];
                    }
                    for (int i = 0; i < weight.Count; i++)
                    {
                        weight[i].Gradient[0] -= ss * result[i].Weight[0];

                    }
                };
                backPropagation.Add(backward);
            }
            return result;
        }

        public void BackPropogation()
        {
            for (var i = backPropagation.Count - 1; i >= 0; i--)
            {
                backPropagation[i](); // tick!
            }
        }
    }
}
