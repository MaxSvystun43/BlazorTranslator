using Seq2SeqModel.Entities;


namespace Seq2SeqModel.Extentions
{
    public class RMSPropOptimazer
    {
        private double decayRate = 0.999;
        private double eps = 1e-8;

        public void Optimize(List<Weights> model, double step_size, double regc, double clipval)
        {
            var numberClipped = 0;
            var totalNumber = 0;
            foreach (var weight in model)
            {
                if (weight == null)
                {
                    continue;
                }
                var temp = weight; // mat ref 
                var s = weight.Cash;
                for (int i = 0, n = temp.Weight.Length; i < n; i++)
                {

                    // rmsprop adaptive learning rate
                    var mdwi = temp.Gradient[i];
                    s[i] = s[i] * decayRate + (1.0 - decayRate)
                        * mdwi * mdwi;

                    // gradient clip
                    if (mdwi > clipval)
                    {
                        mdwi = clipval;
                        numberClipped++;
                    }
                    if (mdwi < -clipval)
                    {
                        mdwi = -clipval;
                        numberClipped++;
                    }
                    totalNumber++;

                    // update (and regularize)
                    temp.Weight[i] += -step_size *
                        mdwi / Math.Sqrt(s[i] + eps) -
                        regc * temp.Weight[i];
                    temp.Gradient[i] = 0; // reset gradients for next iteration
                }

            }
        }
    }
}
