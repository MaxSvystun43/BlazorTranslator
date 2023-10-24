namespace Seq2SeqModel.Entities
{
    public class CostEvent : EventArgs
    {
        public double Cost { get; set; }

        public int Iteration { get; set; }
    }
}
