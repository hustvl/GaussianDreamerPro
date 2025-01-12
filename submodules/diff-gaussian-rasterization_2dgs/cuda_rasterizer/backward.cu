/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__global__ void computeSxyzCUDA(
	int P, int D, int M,
	const int* radii,
	const float3* means,
	const float* shs,
	const bool* clamped,
	const glm::vec4* rots, 
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float* viewmatrix,
	float* dL_dsxyz,
	glm::vec3* dL_dmeans,
	float3* dL_dmean2Ds,
	glm::vec4* dL_drots,
	const glm::vec3* campos,
	float* dL_dcolor,
	float* dL_dsh)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rots[idx];// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Compute 3D world transformation matrix
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	// Compute gradients of loss w.r.t. gaussian center position
	dL_dmeans[idx] = dL_dmeans[idx] * glm::transpose(W);

	glm::mat3 dL_dSxyz_matrix = glm::mat3(
		dL_dsxyz[0], dL_dsxyz[3], dL_dsxyz[6],
		dL_dsxyz[1], dL_dsxyz[4], dL_dsxyz[7],
		dL_dsxyz[2], dL_dsxyz[5], dL_dsxyz[8]
	);

	// Compute gradients of loss w.r.t. rotation matrix
	glm::mat3 dL_dRtranspose = W * glm::transpose(dL_dSxyz_matrix);

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dRtranspose[0][1] - dL_dRtranspose[1][0]) + 2 * y * (dL_dRtranspose[2][0] - dL_dRtranspose[0][2]) + 2 * x * (dL_dRtranspose[1][2] - dL_dRtranspose[2][1]);
	dL_dq.y = 2 * y * (dL_dRtranspose[1][0] + dL_dRtranspose[0][1]) + 2 * z * (dL_dRtranspose[2][0] + dL_dRtranspose[0][2]) + 2 * r * (dL_dRtranspose[1][2] - dL_dRtranspose[2][1]) - 4 * x * (dL_dRtranspose[2][2] + dL_dRtranspose[1][1]);
	dL_dq.z = 2 * x * (dL_dRtranspose[1][0] + dL_dRtranspose[0][1]) + 2 * r * (dL_dRtranspose[2][0] - dL_dRtranspose[0][2]) + 2 * z * (dL_dRtranspose[1][2] + dL_dRtranspose[2][1]) - 4 * y * (dL_dRtranspose[2][2] + dL_dRtranspose[0][0]);
	dL_dq.w = 2 * r * (dL_dRtranspose[0][1] - dL_dRtranspose[1][0]) + 2 * x * (dL_dRtranspose[2][0] + dL_dRtranspose[0][2]) + 2 * y * (dL_dRtranspose[1][2] + dL_dRtranspose[2][1]) - 4 * z * (dL_dRtranspose[1][1] + dL_dRtranspose[0][0]);

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// used for densitify point clouds
	dL_dmean2Ds[idx] = {dL_dmeans[idx].x, dL_dmeans[idx].y, dL_dmeans[idx].z};
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const bool use_integral,
	const float scale_modifier,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ sxyz,
	const float3* __restrict__ scales,
	const float3* __restrict__ means3D_cam,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dout_depths,
	const float* __restrict__ dL_dout_normals,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float3* __restrict__ dL_dscale,
	float3* __restrict__ dL_dmean3D,
	float* __restrict__ dL_dsxyz)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_sx[BLOCK_SIZE];
	__shared__ float3 collected_sy[BLOCK_SIZE];
	__shared__ float3 collected_sz[BLOCK_SIZE];
	__shared__ float3 collected_means3D_cam[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	// for ray rotation
	float W_center = 0.5f * W - 0.5f;
	float H_center = 0.5f * H - 0.5f;
	float focal_x_inv = 1.0f/focal_x;
	float focal_y_inv = 1.0f/focal_y;
	float3 view_dir = { (pixf.x-W_center)*focal_x_inv, (pixf.y-H_center)*focal_y_inv, 1.0f };
	float dep2dist = sqrt(view_dir.x*view_dir.x + view_dir.y*view_dir.y + 1.0f);
	float dist2dep = 1.0f / dep2dist;
	view_dir = { view_dir.x * dist2dep, view_dir.y * dist2dep, view_dir.z * dist2dep };

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float accum_rec_depth = 0;
	float accum_rec_normal[3] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	float dL_dout_d = inside ? dL_dout_depths[pix_id] : 0;
	float dL_dout_n[3];
	if (inside)
		for (int i = 0; i < 3; i++)
			dL_dout_n[i] = dL_dout_normals[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0.0f;
	float last_normal[3] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			int coll_id9 = coll_id * 9;
			collected_sx[block.thread_rank()] = { sxyz[coll_id9 + 0], sxyz[coll_id9 + 1], sxyz[coll_id9 + 2] };
			collected_sy[block.thread_rank()] = { sxyz[coll_id9 + 3], sxyz[coll_id9 + 4], sxyz[coll_id9 + 5] };
			collected_sz[block.thread_rank()] = { sxyz[coll_id9 + 6], sxyz[coll_id9 + 7], sxyz[coll_id9 + 8] };
			collected_means3D_cam[block.thread_rank()] = means3D_cam[coll_id];
			collected_scales[block.thread_rank()] = scales[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute hit point in camera space and gaussian space
			float3 sx = collected_sx[j];
			float3 sy = collected_sy[j];
			float3 sz = collected_sz[j];
			float3 xyz_cam = collected_means3D_cam[j];
			float3 scale = collected_scales[j];
			scale = { scale.x * scale_modifier, scale.y * scale_modifier, scale.z * scale_modifier };
			float n_dot_c = sz.x * xyz_cam.x + sz.y * xyz_cam.y + sz.z * xyz_cam.z;
			float n_dot_view = sz.x * view_dir.x + sz.y * view_dir.y + sz.z * view_dir.z;
			const float n_dot_view_inv = 1.0f / n_dot_view;
			// if the gaussian plane is nearly parallel to the ray, skip
			if (abs(n_dot_view) < 1e-9)
				continue;
			float distance = n_dot_c * n_dot_view_inv;
			// float3 hit_pt = { distance * view_dir.x, distance * view_dir.y, distance * view_dir.z };
			// float3 offset_cam = { hit_pt.x - xyz_cam.x, hit_pt.y - xyz_cam.y, hit_pt.z - xyz_cam.z };
			// float2 uv = { sx.x * offset_cam.x + sx.y * offset_cam.y + sx.z * offset_cam.z, sy.x * offset_cam.x + sy.y * offset_cam.y + sy.z * offset_cam.z };
			// old version of computing hit point. 
			// use the new version to store helpful tmp variables.
			float sx_dot_c = sx.x * xyz_cam.x + sx.y * xyz_cam.y + sx.z * xyz_cam.z;
			float sy_dot_c = sy.x * xyz_cam.x + sy.y * xyz_cam.y + sy.z * xyz_cam.z;
			float sx_dot_view = sx.x * view_dir.x + sx.y * view_dir.y + sx.z * view_dir.z;
			float sy_dot_view = sy.x * view_dir.x + sy.y * view_dir.y + sy.z * view_dir.z;
			float2 uv = { distance * sx_dot_view - sx_dot_c, distance * sy_dot_view - sy_dot_c };
			// 3 sigma
			float2 scale_inv = { 1.0f / scale.x, 1.0f / scale.y };
			const float x_div_sx = uv.x * scale_inv.x;
			const float y_div_sy = uv.y * scale_inv.y;
			if ((abs(x_div_sx) > 3.0f) || (abs(y_div_sy) > 3.0f))
				continue;

			// Sample
			float G = 0.0f;
			const float n_dot_view_inv_abs = abs(n_dot_view_inv);
			float F_x2r = 0.0f, F_y2r = 0.0f;
			float xpr_div_sx = 0.0f, xmr_div_sx = 0.0f, ypr_div_sy = 0.0f, ymr_div_sy = 0.0f;
			const float distance_z_inv = 1.0f / (view_dir.z * distance);
			const float distance_z_inv2 = distance_z_inv * distance_z_inv;
			bool screen_filtering = false;
			if (!use_integral){
				float power = -0.5f * (x_div_sx * x_div_sx + y_div_sy * y_div_sy);
				G = exp(power);
				// // screen filtering
				// float2 xy = collected_xy[j];
				// float2 d = { xy.x - pixf.x, xy.y - pixf.y };
				// float power_screen = -(d.x * d.x + d.y * d.y);
				// if (power > power_screen)
				// 	G = exp(power);
				// else {
				// 	G = exp(power_screen);
				// 	screen_filtering = true;
				// }
			}
			else{
				float radius = 0.25f * distance * view_dir.z * (focal_x_inv + focal_y_inv) * n_dot_view_inv_abs;
				float r_div_sx = radius * scale_inv.x;
				float r_div_sy = radius * scale_inv.y;
				xpr_div_sx = x_div_sx + r_div_sx;
				xmr_div_sx = x_div_sx - r_div_sx;
				ypr_div_sy = y_div_sy + r_div_sy;
				ymr_div_sy = y_div_sy - r_div_sy;
				F_x2r = F_cdf(xpr_div_sx) - F_cdf(xmr_div_sx);
				F_y2r = F_cdf(ypr_div_sy) - F_cdf(ymr_div_sy);
				G = _2pi * abs(n_dot_view) * scale.x * scale.y * F_x2r * F_y2r * focal_x * focal_y * distance_z_inv2;
			}
			
			// // Compute blending values, as before.
			// const float2 xy = collected_xy[j];
			// const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			// const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;
			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			// gradients from regularization
			float3 dLr_dmean3D = { 0 };
			float3 dLr_dn = { 0 };
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			// edit: Depth gradients
			{
				// Update last normal (to be used in the next iteration)
				accum_rec_depth = last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
				last_depth = distance;
				dL_dalpha += (distance - accum_rec_depth) * dL_dout_d; // TODO: Remove the gradients w.r.t alpha?
				// Update gradients from depth
				const float dL_ddistance = dL_dout_d * dchannel_dcolor;
				const float dL_ddistance_div_nv = dL_ddistance * n_dot_view_inv;
				dLr_dmean3D = {dLr_dmean3D.x + sz.x * dL_ddistance_div_nv, 
							   dLr_dmean3D.y + sz.y * dL_ddistance_div_nv, 
							   dLr_dmean3D.z + sz.z * dL_ddistance_div_nv};
				dLr_dn = {dLr_dn.x + dL_ddistance_div_nv * (xyz_cam.x - distance * view_dir.x),
						  dLr_dn.y + dL_ddistance_div_nv * (xyz_cam.y - distance * view_dir.y),
						  dLr_dn.z + dL_ddistance_div_nv * (xyz_cam.z - distance * view_dir.z)};
			}
			// edit: Normal gradients
			const float normal_sign = n_dot_view < 0.0f ? 1.0f : -1.0f;
			const float signed_weight = normal_sign * dchannel_dcolor;
			float n[3] = {sz.x, sz.y, sz.z};
			for (int ch = 0; ch < 3; ch++)
			{
				const float ni = normal_sign * n[ch];
				// Update last normal (to be used in the next iteration)
				accum_rec_normal[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_rec_normal[ch];
				last_normal[ch] = ni;

				const float dL_dchannel = dL_dout_n[ch];
				dL_dalpha += (ni - accum_rec_normal[ch]) * dL_dchannel; // TODO: Remove the gradients w.r.t alpha?
			}
			// Update gradients from normal
			dLr_dn = {dLr_dn.x + signed_weight * dL_dout_n[0], 
					  dLr_dn.y + signed_weight * dL_dout_n[1], 
					  dLr_dn.z + signed_weight * dL_dout_n[2]};

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// // Helpful reusable temporary variables
			// const float dL_dG = con_o.w * dL_dalpha;
			// const float gdx = G * d.x;
			// const float gdy = G * d.y;
			// const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			// const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// // Update gradients w.r.t. 2D mean position of the Gaussian
			// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

			// Helpful reusable temporary variables
			int global_id9 = global_id * 9;
			const float dL_dG = con_o.w * dL_dalpha;
			
			if (!use_integral)
			{
				// if (screen_filtering){
				// 	const float dL_dpower = G * dL_dG;
				// 	const float z_cam_inv = 1.0f / xyz_cam.z;
				// 	const float z_cam_inv2 = z_cam_inv * z_cam_inv;
				// 	float2 xy = collected_xy[j];
				// 	float2 d = { xy.x - pixf.x, xy.y - pixf.y };
				// 	float3 dL_dmean3D_tmp = { dLr_dmean3D.x - 2.0f * d.x * focal_x * z_cam_inv * dL_dpower,
				// 							dLr_dmean3D.y - 2.0f * d.y * focal_y * z_cam_inv * dL_dpower,
				// 							dLr_dmean3D.z + 2.0f * (d.x * focal_x + d.y * focal_y) * z_cam_inv2 * dL_dpower };
				// 	atomicAdd(&dL_dmean3D[global_id].x, dL_dmean3D_tmp.x);
				// 	atomicAdd(&dL_dmean3D[global_id].y, dL_dmean3D_tmp.y);
				// 	atomicAdd(&dL_dmean3D[global_id].z, dL_dmean3D_tmp.z);
				// 	atomicAdd(&dL_dsxyz[global_id9 + 6], dLr_dn.x);
				// 	atomicAdd(&dL_dsxyz[global_id9 + 7], dLr_dn.y);
				// 	atomicAdd(&dL_dsxyz[global_id9 + 8], dLr_dn.z);
				// 	continue;
				// }
				const float dL_dpower = G * dL_dG;
				// gradients w.r.t. uv
				const float dL_du = -uv.x / (scale.x * scale.x) * dL_dpower;
				const float dL_dv = -uv.y / (scale.y * scale.y) * dL_dpower;
				// gradients w.r.t. scale
				float2 dL_dscale_tmp = { uv.x * uv.x / (scale.x * scale.x * scale.x) * dL_dpower, 
										uv.y * uv.y / (scale.y * scale.y * scale.y) * dL_dpower };
				// atomicAdd(&dL_dscale[global_id].x, uv.x * uv.x / (scale.x * scale.x * scale.x) * dL_dpower);
				// atomicAdd(&dL_dscale[global_id].y, uv.y * uv.y / (scale.y * scale.y * scale.y) * dL_dpower);
				// gradients w.r.t. gaussian center
				const float sx_view_div_n_view = sx_dot_view * n_dot_view_inv;
				const float sy_view_div_n_view = sy_dot_view * n_dot_view_inv;
				float3 dL_dmean3D_tmp = { dLr_dmean3D.x + (sz.x * sx_view_div_n_view - sx.x) * dL_du + (sz.x * sy_view_div_n_view - sy.x) * dL_dv,
										  dLr_dmean3D.y + (sz.y * sx_view_div_n_view - sx.y) * dL_du + (sz.y * sy_view_div_n_view - sy.y) * dL_dv,
										  dLr_dmean3D.z + (sz.z * sx_view_div_n_view - sx.z) * dL_du + (sz.z * sy_view_div_n_view - sy.z) * dL_dv };
				// atomicAdd(&dL_dmean3D[global_id].x, (sz.x * sx_view_div_n_view - sx.x) * dL_du + (sz.x * sy_view_div_n_view - sy.x) * dL_dv);
				// atomicAdd(&dL_dmean3D[global_id].y, (sz.y * sx_view_div_n_view - sx.y) * dL_du + (sz.y * sy_view_div_n_view - sy.y) * dL_dv);
				// atomicAdd(&dL_dmean3D[global_id].z, (sz.z * sx_view_div_n_view - sx.z) * dL_du + (sz.z * sy_view_div_n_view - sy.z) * dL_dv);
				// gradients w.r.t. rotation
				const float3 offset_cam = { distance * view_dir.x - xyz_cam.x, distance * view_dir.y - xyz_cam.y, distance * view_dir.z - xyz_cam.z };
				const float fix_coeff = sx_view_div_n_view * dL_du + sy_view_div_n_view * dL_dv;
				float dL_dsxyz_tmp[9] = { offset_cam.x * dL_du, offset_cam.y * dL_du, offset_cam.z * dL_du,
										  offset_cam.x * dL_dv, offset_cam.y * dL_dv, offset_cam.z * dL_dv,
										  dLr_dn.x + fix_coeff * (xyz_cam.x - distance * view_dir.x),
										  dLr_dn.y + fix_coeff * (xyz_cam.y - distance * view_dir.y),
										  dLr_dn.z + fix_coeff * (xyz_cam.z - distance * view_dir.z) };
				// atomicAdd(&dL_dsxyz[global_id9], offset_cam.x * dL_du);
				// atomicAdd(&dL_dsxyz[global_id9 + 1], offset_cam.y * dL_du);
				// atomicAdd(&dL_dsxyz[global_id9 + 2], offset_cam.z * dL_du);
				// atomicAdd(&dL_dsxyz[global_id9 + 3], offset_cam.x * dL_dv);
				// atomicAdd(&dL_dsxyz[global_id9 + 4], offset_cam.y * dL_dv);
				// atomicAdd(&dL_dsxyz[global_id9 + 5], offset_cam.z * dL_dv);
				// atomicAdd(&dL_dsxyz[global_id9 + 6], fix_coeff * xyz_cam.x * (n_dot_view - sz.x * view_dir.x));
				// atomicAdd(&dL_dsxyz[global_id9 + 7], fix_coeff * xyz_cam.y * (n_dot_view - sz.y * view_dir.y));
				// atomicAdd(&dL_dsxyz[global_id9 + 8], fix_coeff * xyz_cam.z * (n_dot_view - sz.z * view_dir.z));

				// atomic add
				atomicAdd(&dL_dscale[global_id].x, dL_dscale_tmp.x);
				atomicAdd(&dL_dscale[global_id].y, dL_dscale_tmp.y);
				atomicAdd(&dL_dmean3D[global_id].x, dL_dmean3D_tmp.x);
				atomicAdd(&dL_dmean3D[global_id].y, dL_dmean3D_tmp.y);
				atomicAdd(&dL_dmean3D[global_id].z, dL_dmean3D_tmp.z);
				atomicAdd(&dL_dsxyz[global_id9], dL_dsxyz_tmp[0]);
				atomicAdd(&dL_dsxyz[global_id9 + 1], dL_dsxyz_tmp[1]);
				atomicAdd(&dL_dsxyz[global_id9 + 2], dL_dsxyz_tmp[2]);
				atomicAdd(&dL_dsxyz[global_id9 + 3], dL_dsxyz_tmp[3]);
				atomicAdd(&dL_dsxyz[global_id9 + 4], dL_dsxyz_tmp[4]);
				atomicAdd(&dL_dsxyz[global_id9 + 5], dL_dsxyz_tmp[5]);
				atomicAdd(&dL_dsxyz[global_id9 + 6], dL_dsxyz_tmp[6]);
				atomicAdd(&dL_dsxyz[global_id9 + 7], dL_dsxyz_tmp[7]);
				atomicAdd(&dL_dsxyz[global_id9 + 8], dL_dsxyz_tmp[8]);
			}
			else {
				const float f_xplusr = F_pdf(xpr_div_sx);
				const float f_xminusr = F_pdf(xmr_div_sx);
				const float f_yplusr = F_pdf(ypr_div_sy);
				const float f_yminusr = F_pdf(ymr_div_sy);
				const float dL_dG_tmp = dL_dG * _2pi * focal_x * focal_y * distance_z_inv2;
				const float dL_dG_coeff = dL_dG_tmp * abs(n_dot_view);
				const float dL_dG_coeff2 = dL_dG_tmp * scale.x * scale.y * F_x2r * F_y2r;
				const float dL_dG_coeff3 = dL_dG_coeff2 * 2.0f * normal_sign * view_dir.z * distance_z_inv;
				// gradients w.r.t. scale
				float2 dL_dscale_tmp = { dL_dG_coeff * scale.y * F_y2r * (F_x2r + f_xminusr * xmr_div_sx - f_xplusr * xpr_div_sx),
										dL_dG_coeff * scale.x * F_x2r * (F_y2r + f_yminusr * ymr_div_sy - f_yplusr * ypr_div_sy) }; 
				// gradients w.r.t. uvr
				const float dL_du = dL_dG_coeff * scale.y * F_y2r * (f_xplusr - f_xminusr);
				const float dL_dv = dL_dG_coeff * scale.x * F_x2r * (f_yplusr - f_yminusr);
				const float dL_dr = dL_dG_coeff * (scale.y * F_y2r * (f_xplusr + f_xminusr) + scale.x * F_x2r * (f_yplusr + f_yminusr));
				// gradients w.r.t. gaussian center
				const float sx_view_div_n_view = sx_dot_view * n_dot_view_inv;
				const float sy_view_div_n_view = sy_dot_view * n_dot_view_inv;
				const float n_dot_view_inv2 = n_dot_view_inv * n_dot_view_inv;
				const float dL_dr_coeff = 0.25f * dL_dr * view_dir.z * (focal_x_inv + focal_y_inv) * n_dot_view_inv2;
				float3 dL_dmean3D_tmp = { dLr_dmean3D.x + (sz.x * sx_view_div_n_view - sx.x) * dL_du + (sz.x * sy_view_div_n_view - sy.x) * dL_dv + sz.x * (dL_dr_coeff + dL_dG_coeff3),
										  dLr_dmean3D.y + (sz.y * sx_view_div_n_view - sx.y) * dL_du + (sz.y * sy_view_div_n_view - sy.y) * dL_dv + sz.y * (dL_dr_coeff + dL_dG_coeff3),
										  dLr_dmean3D.z + (sz.z * sx_view_div_n_view - sx.z) * dL_du + (sz.z * sy_view_div_n_view - sy.z) * dL_dv + sz.z * (dL_dr_coeff + dL_dG_coeff3) };
				// gradients w.r.t. rotation
				const float3 offset_cam = { distance * view_dir.x - xyz_cam.x, distance * view_dir.y - xyz_cam.y, distance * view_dir.z - xyz_cam.z };
				const float fix_coeff = n_dot_view_inv * (sx_view_div_n_view * dL_du + sy_view_div_n_view * dL_dv);

				float dL_dsxyz_tmp[9] = { offset_cam.x * dL_du, offset_cam.y * dL_du, offset_cam.z * dL_du,
										  offset_cam.x * dL_dv, offset_cam.y * dL_dv, offset_cam.z * dL_dv,
										  dLr_dn.x + fix_coeff * xyz_cam.x * (n_dot_view - sz.x * view_dir.x) + (-normal_sign) * (dL_dG_coeff2 * view_dir.x + dL_dr_coeff * (xyz_cam.x - 2.0f * view_dir.x * distance)) + dL_dG_coeff3 * (xyz_cam.x * n_dot_view - view_dir.x * n_dot_c),
										  dLr_dn.y + fix_coeff * xyz_cam.y * (n_dot_view - sz.y * view_dir.y) + (-normal_sign) * (dL_dG_coeff2 * view_dir.y + dL_dr_coeff * (xyz_cam.y - 2.0f * view_dir.y * distance)) + dL_dG_coeff3 * (xyz_cam.y * n_dot_view - view_dir.y * n_dot_c),
										  dLr_dn.z + fix_coeff * xyz_cam.z * (n_dot_view - sz.z * view_dir.z) + (-normal_sign) * (dL_dG_coeff2 * view_dir.z + dL_dr_coeff * (xyz_cam.z - 2.0f * view_dir.z * distance)) + dL_dG_coeff3 * (xyz_cam.z * n_dot_view - view_dir.z * n_dot_c) };
				// atomic add
				atomicAdd(&dL_dscale[global_id].x, dL_dscale_tmp.x);
				atomicAdd(&dL_dscale[global_id].y, dL_dscale_tmp.y);
				atomicAdd(&dL_dmean3D[global_id].x, dL_dmean3D_tmp.x);
				atomicAdd(&dL_dmean3D[global_id].y, dL_dmean3D_tmp.y);
				atomicAdd(&dL_dmean3D[global_id].z, dL_dmean3D_tmp.z);
				atomicAdd(&dL_dsxyz[global_id9], dL_dsxyz_tmp[0]);
				atomicAdd(&dL_dsxyz[global_id9 + 1], dL_dsxyz_tmp[1]);
				atomicAdd(&dL_dsxyz[global_id9 + 2], dL_dsxyz_tmp[2]);
				atomicAdd(&dL_dsxyz[global_id9 + 3], dL_dsxyz_tmp[3]);
				atomicAdd(&dL_dsxyz[global_id9 + 4], dL_dsxyz_tmp[4]);
				atomicAdd(&dL_dsxyz[global_id9 + 5], dL_dsxyz_tmp[5]);
				atomicAdd(&dL_dsxyz[global_id9 + 6], dL_dsxyz_tmp[6]);
				atomicAdd(&dL_dsxyz[global_id9 + 7], dL_dsxyz_tmp[7]);
				atomicAdd(&dL_dsxyz[global_id9 + 8], dL_dsxyz_tmp[8]);
			}
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dsxyz)
{
	// // Propagate gradients for the path of 2D conic matrix computation. 
	// // Somewhat long, thus it is its own kernel rather than being part of 
	// // "preprocess". When done, loss gradient w.r.t. 3D means has been
	// // modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
	// 	P,
	// 	means3D,
	// 	radii,
	// 	cov3Ds,
	// 	focal_x,
	// 	focal_y,
	// 	tan_fovx,
	// 	tan_fovy,
	// 	viewmatrix,
	// 	dL_dconic,
	// 	(float3*)dL_dmean3D,
	// 	dL_dcov3D);

	// // Propagate gradients for remaining steps: finish 3D mean gradients,
	// // propagate color gradients to SH (if desireD), propagate 3D covariance
	// // matrix gradients to scale and rotation.
	// preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
	// 	P, D, M,
	// 	(float3*)means3D,
	// 	radii,
	// 	shs,
	// 	clamped,
	// 	(glm::vec3*)scales,
	// 	(glm::vec4*)rotations,
	// 	scale_modifier,
	// 	projmatrix,
	// 	campos,
	// 	(float3*)dL_dmean2D,
	// 	(glm::vec3*)dL_dmean3D,
	// 	dL_dcolor,
	// 	dL_dcov3D,
	// 	dL_dsh,
	// 	dL_dscale,
	// 	dL_drot);

	computeSxyzCUDA << <(P + 255) / 256, 256 >> > (
		P, D, M,
		radii,
		(float3*)means3D,
		shs,
		clamped,
		(glm::vec4*)rotations,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		viewmatrix,
		dL_dsxyz,
		(glm::vec3*)dL_dmean3D,
		(float3*)dL_dmean2D,
		dL_drot,
		campos,
		dL_dcolor,
		dL_dsh);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const bool use_integral,
	const float scale_modifier,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* sxyz,
	const float3* scales,
	const float3* means3D_cam,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dout_depth,
	const float* dL_dout_normal,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float3* dL_dscale,
	float3* dL_dmean3D,
	float* dL_dsxyz)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		use_integral,
		scale_modifier,
		bg_color,
		means2D,
		conic_opacity,
		sxyz,
		scales,
		means3D_cam,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dout_depth,
		dL_dout_normal,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dscale,
		dL_dmean3D,
		dL_dsxyz
		);
}