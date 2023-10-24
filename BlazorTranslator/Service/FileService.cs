using Seq2SeqModel.Entities;
using IHostingEnvironment = Microsoft.AspNetCore.Hosting.IHostingEnvironment;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;

namespace TranslatorBlazor.Services
{
    public class FileService
    {
        private IHostingEnvironment Environment;
        private int FileNamePosition = 0;
        private string FolderName = "Models";

        public FileService(IHostingEnvironment env)
        {
            Environment = env;
        }
        public async void AddFileModelData(ModelData model, string name)
        {
            string path = Path.Combine(Environment.WebRootPath, FolderName);
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
            BinaryFormatter bf = new BinaryFormatter();
            using FileStream fileStream = new FileStream($"{path}/{name}.bin", FileMode.OpenOrCreate, FileAccess.Write);
            bf.Serialize(fileStream, model);
        }

        public ModelData GetFileModelData(string name)
        {
            string path = Path.Combine(Environment.WebRootPath, FolderName);
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
            BinaryFormatter bf = new BinaryFormatter();
            using FileStream fileStream = new FileStream($"{path}/{name}.bin", FileMode.Open, FileAccess.Read);
            ModelData result = bf.Deserialize(fileStream) as ModelData;

            return result;
        }

        public string GetAllTranslations()
        {
            string path = Path.Combine(Environment.WebRootPath, "Models");
            if (!Directory.Exists(path))
            {
                Directory.CreateDirectory(path);
            }
            DirectoryInfo d = new DirectoryInfo(path);

            FileInfo[] Files = d.GetFiles("*.bin"); //Getting bin files
            string str = "";

            foreach (FileInfo file in Files)
            {
                str = str + "\n" + file.Name.Split(".")[FileNamePosition];
            }
            return str;
        }
    }
}
