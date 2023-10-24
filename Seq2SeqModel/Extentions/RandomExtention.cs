using System;

namespace Seq2SeqModel.Extentions
{
    public static class RandomExtention
    {
        //return old value or generate new one
        public static bool ReturnPrev { get; set; }
        //old value
        public static double Value { get; set; }

        private static Random randomGenerator = new Random(3);

        public static double GaussRandom()
        {
            if (ReturnPrev)
            {
                ReturnPrev = false;
                return Value;
            }
            var u = 2 * randomGenerator.NextDouble() - 1;
            var v = 2 * randomGenerator.NextDouble() - 1;
            var r = (u * u) + (v * v);

            if (r == 0 || r > 1)
                return GaussRandom();
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            Value = v * c;
            ReturnPrev = true;
            return u * c;
        }

        public static double NormalRandom(double mu, double standardDeviation)
        {
            return mu + GaussRandom() * standardDeviation;
        }
    }
}
