using Seq2SeqModel.Extentions;

namespace Seq2SeqModel.Entities
{
    [Serializable]
    public class Weights
    {
        //Represents the number of rows in the weight matrix.
        public int Rows { get; set; }
        //Represents the number of columns in the weight matrix.
        public int Columns { get; set; }
        //An array that holds the values of the weights.
        public double[] Weight { get; set; }
        //An array that holds the values of the gradients.
        public double[] Gradient { get; set; }
        //An array that holds temporary values.
        public double[] Cash { get; set; }

        /// <summary>
        /// Initializes a new instance of the Weights class with the specified number of rows and columns. 
        /// The <paramref name="normal"/> determines whether the weights should be initialized with random values 
        /// <code>using RandomExtention.NormalRandom</code> or a constant scale.
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="columns"></param>
        /// <param name="normal"></param>
        public Weights(int rows, int columns, bool normal = false)
        {
            Rows = rows;
            Columns = columns;
            var n = rows * columns;
            Weight = new double[n];
            Gradient = new double[n];
            Cash = new double[n];

            var scale = Math.Sqrt(1.0 / (rows * columns));
            if (normal)
            {
                scale = 0.08;
            }
            for (int i = 0; i < n; i++)
            {
                Weight[i] = RandomExtention.NormalRandom(0.0, scale);
            }

        }
        /// <summary>
        /// Initializes a new instance of the Weights class with the specified number of rows and columns. 
        /// The weights are initialized with the constant value <paramref name="value"/>.
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="columns"></param>
        /// <param name="value"></param>
        public Weights(int rows, int columns, double value)
        {
            Rows = rows;
            Columns = columns;
            var n = rows * columns;
            Weight = new double[n];
            Gradient = new double[n];

            Cash = new double[n];
            for (int i = 0; i < n; i++)
            {
                Weight[i] = value;
            }

        }

        /// <summary>
        /// Retrieves the weight value at the specified <paramref name="row"/> and <paramref name="column"/>.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public double GetWeight(int row, int column)
        {
            var ix = (Columns * row) + column;
            return Weight[ix];
        }

        /// <summary>
        /// Sets the weight value at the specified <paramref name="row"/> and <paramref name="column"/> to the given value <paramref name="value"/>.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="value"></param>
        public void SetWeight(int row, int column, double value)
        {
            var ix = (Columns * row) + column;
            Weight[ix] = value;
        }

        /// <summary>
        /// Adds the given value <paramref name="value"/> to the weight at the specified <paramref name="row"/> and <paramref name="column"/>.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="value"></param>
        public void AddToWeight(int row, int column, double value)
        {
            var ix = (Columns * row) + column;
            Weight[ix] += value;
        }

        /// <summary>
        /// Retrieves the gradient value at the specified <paramref name="row"/> and <paramref name="column"/>.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <returns></returns>
        public double GetGradient(int row, int column)
        {
            var ix = (Columns * row) + column;
            return Gradient[ix];
        }

        /// <summary>
        /// Adds the given value <paramref name="value"/> to the gradient value at the specified <paramref name="row"/> and <paramref name="column"/>.
        /// </summary>
        /// <param name="row"></param>
        /// <param name="column"></param>
        /// <param name="value"></param>
        public void AddGradient(int row, int column, double value)
        {
            var ix = (Columns * row) + column;
            Gradient[ix] += value;
        }

        /// <summary>
        /// Creates a deep copy of the current <c>Weight</c> object, including the weight values.
        /// </summary>
        /// <returns></returns>
        public Weights CloneData()
        {
            var result = new Weights(Rows, Columns, 0.0);
            var lengthOfWeight = Weight.Length;
            for (int i = 0; i < lengthOfWeight; i++)
            {
                result.Weight[i] = Weight[i];
            }
            return result;
        }

    }
}
