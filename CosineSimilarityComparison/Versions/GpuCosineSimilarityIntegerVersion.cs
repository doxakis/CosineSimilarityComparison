using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace CosineSimilarityComparison.Versions
{
    public class GpuCosineSimilarityIntegerVersion
    {
		public static double[][] ComputeDistances(int[][] dataSet)
		{
			int numSample = dataSet.Length;
			int dim = dataSet[0].Length;

			int[] dataset = new int[numSample * dim];
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
						var kernel = accelerator.LoadAutoGroupedStreamKernel<
							Index, ArrayView2D<int>, ArrayView2D<double>>(CosineSimilarityKernel);

						using (var gpuDistances = accelerator.Allocate<double>(numSample * numSample))
						using (var gpuDataset = accelerator.Allocate<int>(numSample * dim))
						{
							gpuDataset.CopyFrom(dataset, 0, 0, dataset.Length);

							// Launch buffer.Length many threads and pass a view to buffer
							// Note that the kernel launch does not involve any boxing
							var a = gpuDataset.As2DView(numSample, dim);
							var b = gpuDistances.As2DView(numSample, numSample);
							kernel(numSample * numSample, a, b);

							// Wait for the kernel to finish...
							accelerator.Synchronize();

							// Resolve and verify data
							var data = gpuDistances.GetAsArray();

							double[][] distancesVector = new double[numSample][];
							for (int i = 0; i < numSample; i++)
							{
								distancesVector[i] = new double[numSample];
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
			ArrayView2D<int> dataset,
			ArrayView2D<double> distances)
		{
			int rows = dataset.Rows;
			int i = index / rows;
			int j = index % rows;

			if (i < j) return;

			double dotProduct = 0;
			double magnitudeOne = 0;
			double magnitudeTwo = 0;
			for (int k = 0; k < dataset.Columns; k++)
			{
				dotProduct += (dataset[i, k] * dataset[j, k]);
				magnitudeOne += (dataset[i, k] * dataset[i, k]);
				magnitudeTwo += (dataset[j, k] * dataset[j, k]);
			}
			double distance = double.NaN;
			double divisor = GPUMath.Sqrt(magnitudeOne * magnitudeTwo);
			if (divisor != 0)
			{
				distance = GPUMath.Max(0, 1 - (dotProduct / divisor));
			}
			distances[i, j] = distance;
			distances[j, i] = distance;
		}

	}
}
