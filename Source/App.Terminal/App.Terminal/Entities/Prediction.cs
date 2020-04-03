using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace App.Terminal.Entities
{
    public class Prediction
    {
        [ColumnName("Score")]
        public float Price { get; set; }
    }
}
