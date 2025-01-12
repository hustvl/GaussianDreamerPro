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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
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

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Compute Sxyz in world space for each Gaussian.
__device__ void computeSxyz(const glm::vec3 scale, float mod, const glm::vec4 rot, float* sxyz, const float* viewmatrix)
{
	// Create scaling matrix
	// glm::mat3 S = glm::mat3(1.0f);
	// S[0][0] = mod * scale.x;
	// S[1][1] = mod * scale.y;
	// S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
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

	// Compute final Sxyz matrix: Sxyz = W @ R @ I
	glm::mat3 Sxyz_matrix = R * W;

	// Store Sxyz matrix in row-major order
	sxyz[0] = Sxyz_matrix[0][0];
	sxyz[1] = Sxyz_matrix[1][0];
	sxyz[2] = Sxyz_matrix[2][0];
	sxyz[3] = Sxyz_matrix[0][1];
	sxyz[4] = Sxyz_matrix[1][1];
	sxyz[5] = Sxyz_matrix[2][1];
	sxyz[6] = Sxyz_matrix[0][2];
	sxyz[7] = Sxyz_matrix[1][2];
	sxyz[8] = Sxyz_matrix[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float3* means3D_cam,
	float* depths,
	float* cov3Ds,
	float* sxyzs,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	means3D_cam[idx] = p_view;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute Sxyz in world coordinates
	computeSxyz(scales[idx], scale_modifier, rotations[idx], sxyzs + idx * 9, viewmatrix);

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const bool use_integral,
	const float scale_modifier,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ sxyz,
	const float3* __restrict__ scales,
	const float3* __restrict__ means3D_cam,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	float* __restrict__ out_normal,
	float* __restrict__ out_accum)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float3 collected_sx[BLOCK_SIZE];
	__shared__ float3 collected_sy[BLOCK_SIZE];
	__shared__ float3 collected_sz[BLOCK_SIZE];
	__shared__ float3 collected_means3D_cam[BLOCK_SIZE];
	__shared__ float3 collected_scales[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float Depth = 0.0f;
	float preDepth = 0.0f;
	float Normal[3] = { 0 };
	// for ray rotation
	float W_center = 0.5f * W - 0.5f;
	float H_center = 0.5f * H - 0.5f;
	float focal_x_inv = 1.0f/focal_x;
	float focal_y_inv = 1.0f/focal_y;
	float3 view_dir = { (pixf.x-W_center)*focal_x_inv, (pixf.y-H_center)*focal_y_inv, 1.0f };
	float dep2dist = sqrt(view_dir.x*view_dir.x + view_dir.y*view_dir.y + 1.0f);
	float dist2dep = 1.0f / dep2dist;
	view_dir = { view_dir.x * dist2dep, view_dir.y * dist2dep, view_dir.z * dist2dep };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			int coll_id9 = coll_id * 9;
			collected_sx[block.thread_rank()] = { sxyz[coll_id9 + 0], sxyz[coll_id9 + 1], sxyz[coll_id9 + 2] };
			collected_sy[block.thread_rank()] = { sxyz[coll_id9 + 3], sxyz[coll_id9 + 4], sxyz[coll_id9 + 5] };
			collected_sz[block.thread_rank()] = { sxyz[coll_id9 + 6], sxyz[coll_id9 + 7], sxyz[coll_id9 + 8] };
			collected_means3D_cam[block.thread_rank()] = means3D_cam[coll_id];
			collected_scales[block.thread_rank()] = scales[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Compute hit point in camera space and gaussian space
			float3 sx = collected_sx[j];
			float3 sy = collected_sy[j];
			float3 sz = collected_sz[j];
			float3 xyz_cam = collected_means3D_cam[j];
			float3 scale = collected_scales[j];
			scale = { scale.x * scale_modifier, scale.y * scale_modifier, scale.z * scale_modifier }; // TODO: do this in process
			float n_dot_c = sz.x * xyz_cam.x + sz.y * xyz_cam.y + sz.z * xyz_cam.z; // TODO: do this in process
			float n_dot_view = sz.x * view_dir.x + sz.y * view_dir.y + sz.z * view_dir.z;
			// if the gaussian plane is nearly parallel to the ray, skip
			if (abs(n_dot_view) < 1e-9)
				continue;
			const float n_dot_view_inv = 1.0f / n_dot_view;
			float distance = n_dot_c * n_dot_view_inv;
			float3 hit_pt = { distance * view_dir.x, distance * view_dir.y, distance * view_dir.z };
			float3 offset_cam = { hit_pt.x - xyz_cam.x, hit_pt.y - xyz_cam.y, hit_pt.z - xyz_cam.z };
			float2 uv = { sx.x * offset_cam.x + sx.y * offset_cam.y + sx.z * offset_cam.z, sy.x * offset_cam.x + sy.y * offset_cam.y + sy.z * offset_cam.z };
			// 3 sigma
			float2 scale_inv = { 1.0f / scale.x, 1.0f / scale.y };
			const float x_div_sx = uv.x * scale_inv.x;
			const float y_div_sy = uv.y * scale_inv.y;
			if ((abs(x_div_sx) > 3.0f) || (abs(y_div_sy) > 3.0f))
				continue;

			float G = 0.0f;
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
			else {
				float radius = 0.25f * hit_pt.z * (focal_x_inv + focal_y_inv) * abs(n_dot_view_inv);
				const float r_div_sx = radius * scale_inv.x;
				const float r_div_sy = radius * scale_inv.y;
				const float z_inv2 = 1.0f / (hit_pt.z * hit_pt.z);
				G = _2pi * abs(n_dot_view) * scale.x * scale.y * focal_x * focal_y * z_inv2
					* (F_cdf(x_div_sx + r_div_sx) - F_cdf(x_div_sx - r_div_sx))
					* (F_cdf(y_div_sy + r_div_sy) - F_cdf(y_div_sy - r_div_sy));
			}
			
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			// float2 xy = collected_xy[j];
			// float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			float weight = alpha * T;
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * weight;
			Depth += distance * weight; // weighted depth
			// midpoint depth
			// if (preDepth == 0.0f)
			// 	preDepth = distance;
			// if (T > 0.5f && test_T <= 0.5f)
			// 	Depth = preDepth + (distance - preDepth) * (T - 0.5f) / (T - test_T);
			// preDepth = distance;
			float normal_sign = n_dot_view < 0.0f ? 1.0f : -1.0f;
			float signed_weight = normal_sign * weight;
			Normal[0] = Normal[0] + signed_weight * sz.x;
			Normal[1] = Normal[1] + signed_weight * sz.y;
			Normal[2] = Normal[2] + signed_weight * sz.z;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		// TODO: normalize according to final_T?
		out_depth[pix_id] = Depth;
		for (int ch = 0; ch < 3; ch++)
			out_normal[ch * H * W + pix_id] = Normal[ch];
		out_accum[pix_id] = 1.0f - T;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float focal_x, const float focal_y,
	const bool use_integral,
	const float scale_modifier,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	const float* sxyz,
	const float3* scales,
	const float3* means3D_cam,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	float* out_normal,
	float* out_accum)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		use_integral,
		scale_modifier,
		means2D,
		colors,
		conic_opacity,
		sxyz,
		scales,
		means3D_cam,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		out_normal,
		out_accum);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float3* means3D_cam,
	float* depths,
	float* cov3Ds,
	float* sxyz,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		means3D_cam,
		depths,
		cov3Ds,
		sxyz,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}