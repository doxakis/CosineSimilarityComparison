using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace CosineSimilarityComparison.Versions
{
    public class GpuCosineSimilarityFloatVersion
    {
		public static float[][] ComputeDistances(float[][] dataSet)
		{
			int numSample = dataSet.Length;
			int dim = dataSet[0].Length;

			float[] dataset = new float[numSample * dim];
			for (int i = 0; i < numSample; i++)
				for (int j = 0; j < dim; j++)
					dataset[i + j * numSample] = dataSet[i][j];

			foreach (var acceleratorId in Accelerator.Accelerators)
			{
				if (acceleratorId.AcceleratorType == AcceleratorType.Cuda)
				{
					// We will use the first CUDA device.

					using (var context = new Context())
					using (var accelerator = Accelerator.Create(context, acceleratorId))
					{
						Action<Index, ArrayView2D<float>, ArrayView2D<float>> kernel;
						{
							var watch = Stopwatch.StartNew();
							kernel = accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<float>, ArrayView2D<float>>(CosineSimilarityKernel);
							//Console.WriteLine("    accelerator.LoadAutoGroupedStreamKernel(): " + watch.Elapsed);
						}

						using (var gpuDistances = accelerator.Allocate<float>(numSample * numSample))
						using (var gpuDataset = accelerator.Allocate<float>(numSample * dim))
						{
							{
								var watch = Stopwatch.StartNew();
								gpuDataset.CopyFrom(dataset, 0, 0, dataset.Length);
								//Console.WriteLine("    gpuDataset.CopyFrom(): " + watch.Elapsed);
							}

							// Launch buffer.Length many threads and pass a view to buffer
							// Note that the kernel launch does not involve any boxing
							var a = gpuDataset.As2DView(numSample, dim);
							var b = gpuDistances.As2DView(numSample, numSample);
							kernel(numSample * numSample, a, b);

							{
								var watch = Stopwatch.StartNew();
								// Wait for the kernel to finish...
								accelerator.Synchronize();
								//Console.WriteLine("    accelerator.Synchronize(): " + watch.Elapsed);
							}

							// Resolve and verify data
							var data = gpuDistances.GetAsArray();

							float[][] distancesVector = new float[numSample][];
							for (int i = 0; i < numSample; i++)
							{
								distancesVector[i] = new float[numSample];
								for (int j = 0; j < numSample; j++)
								{
									distancesVector[i][j] = data[i + j * numSample];
								}
							}
							return distancesVector;
						}
					}
				}
			}
			throw new Exception("No GPU found.");
		}

		public static void CosineSimilarityKernel(
			Index index,
			ArrayView2D<float> dataset,
			ArrayView2D<float> distances)
		{
			int rows = dataset.Rows;
			int i = index / rows;
			int j = index % rows;

			if (i < j) return;

			float dotProduct = 0;
			float magnitudeOne = 0;
			float magnitudeTwo = 0;
			for (int k = 0; k < dataset.Columns; k++)
			{
				dotProduct += (dataset[i, k] * dataset[j, k]);
				magnitudeOne += (dataset[i, k] * dataset[i, k]);
				magnitudeTwo += (dataset[j, k] * dataset[j, k]);
			}
			float distance = float.NaN;
			float divisor = GPUMath.Sqrt(magnitudeOne * magnitudeTwo);
			if (divisor != 0)
			{
				distance = GPUMath.Max(0, 1 - (dotProduct / divisor));
			}
			distances[i, j] = distance;
			distances[j, i] = distance;
		}

	}
}
