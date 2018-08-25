using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace CosineSimilarityComparison.Versions
{
    public class GpuCosineSimilarityIntegerVersionCacheKernel : IDisposable
    {
		public bool IsInitialized = false;
		public Context Context;
		public Accelerator Accelerator;
		public Action<Index, ArrayView2D<int>, ArrayView2D<double>> Kernel;

		public void Dispose()
		{
			if (!IsInitialized)
			{
				throw new InvalidOperationException("Not initialized.");
			}

			IsInitialized = false;
			Accelerator.Dispose();
			Context.Dispose();
		}

		public void Init()
		{
			if (IsInitialized)
			{
				throw new InvalidOperationException("Already initialized.");
			}

			IsInitialized = true;
			foreach (var acceleratorId in Accelerator.Accelerators)
			{
				if (acceleratorId.AcceleratorType == AcceleratorType.Cuda)
				{
					// We will use the first CUDA device.

					Context = new Context();
					Accelerator = Accelerator.Create(Context, acceleratorId);
					Kernel = Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<int>, ArrayView2D<double>>(CosineSimilarityKernel);
					return;
				}
			}
			throw new Exception("No GPU found.");
		}

		public double[][] ComputeDistances(int[][] dataSet)
		{
			if (!IsInitialized)
			{
				throw new InvalidOperationException("Not initialized.");
			}

			int numSample = dataSet.Length;
			int dim = dataSet[0].Length;

			int[] dataset = new int[numSample * dim];
			for (int i = 0; i < numSample; i++)
				for (int j = 0; j < dim; j++)
					dataset[i + j * numSample] = dataSet[i][j];

			using (var gpuDistances = Accelerator.Allocate<double>(numSample * numSample))
			using (var gpuDataset = Accelerator.Allocate<int>(numSample * dim))
			{
				gpuDataset.CopyFrom(dataset, 0, 0, dataset.Length);

				// Launch buffer.Length many threads and pass a view to buffer
				// Note that the kernel launch does not involve any boxing
				var a = gpuDataset.As2DView(numSample, dim);
				var b = gpuDistances.As2DView(numSample, numSample);
				Kernel(numSample * numSample, a, b);

				// Wait for the kernel to finish...
				Accelerator.Synchronize();

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
