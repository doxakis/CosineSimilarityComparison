using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace CosineSimilarityComparison.Versions
{
    public class GpuCosineSimilarityFloatVersionCacheKernel : IDisposable
    {
		public bool IsInitialized = false;
		public Context Context;
		public Accelerator Accelerator;
		public Action<Index, ArrayView2D<float>, ArrayView2D<float>> Kernel;

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
					Kernel = Accelerator.LoadAutoGroupedStreamKernel<Index, ArrayView2D<float>, ArrayView2D<float>>(CosineSimilarityKernel);
					return;
				}
			}
			throw new Exception("No GPU found.");
		}

		public float[][] ComputeDistances(float[][] dataSet)
		{
			if (!IsInitialized)
			{
				throw new InvalidOperationException("Not initialized.");
			}

			int numSample = dataSet.Length;
			int dim = dataSet[0].Length;

			float[] dataset = new float[numSample * dim];
			for (int i = 0; i < numSample; i++)
				for (int j = 0; j < dim; j++)
					dataset[i + j * numSample] = dataSet[i][j];

			using (var gpuDistances = Accelerator.Allocate<float>(numSample * numSample))
			using (var gpuDataset = Accelerator.Allocate<float>(numSample * dim))
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
