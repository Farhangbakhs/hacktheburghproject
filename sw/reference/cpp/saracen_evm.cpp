// saracen_evm.cpp - saracen/magnification's per-frame EVM as a DLL with a C API.
//
// The processing below is copied from third_party/saracen/main.cpp (MIT licence,
// Copyright (c) 2018 Arran Walker; see third_party/saracen/LICENSE). Changes:
//   * the input/main/output threads and VideoCapture/imshow are removed; the
//     caller (Python dashboard via ctypes) pushes one BGR frame at a time and
//     gets the magnified BGR frame back, so it can mask it to the face;
//   * state lives in a struct instead of main()'s locals;
//   * settings can be changed live (evm_set) and state reset (evm_reset).
// The colour conversion, Laplacian decomposition, per-level IIR filter,
// spatial-wavelength alpha clamp, per-level worker threads, reconstruction,
// chroma attenuation and output conversion are unchanged.

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

#define EVM_API extern "C" __declspec(dllexport)

struct settings
{
	int levels = 5;
	double alpha = 20.0;
	double lambda_c = 20.0;
	double cutoff_frequency_high = 0.4;
	double cutoff_frequency_low = 0.05;
	double chrom_attenuation = 0.1;
	double exaggeration_factor = 2.0;
};

struct evm_state
{
	settings s;
	int frame_num = 0;
	int w = 0, h = 0;
	std::vector<cv::Mat> low_pass1;
	std::vector<cv::Mat> low_pass2;
	std::vector<cv::Mat> filtered;
};

EVM_API void *evm_create(int levels, double alpha, double lambda_c, double cutoff_high,
						 double cutoff_low, double chrom_attenuation, double exaggeration)
{
	auto *st = new evm_state();
	st->s = settings{levels, alpha, lambda_c, cutoff_high, cutoff_low, chrom_attenuation, exaggeration};
	return st;
}

EVM_API void evm_set(void *handle, double alpha, double lambda_c, double cutoff_high,
					 double cutoff_low, double chrom_attenuation, double exaggeration)
{
	auto *st = static_cast<evm_state *>(handle);
	st->s.alpha = alpha;
	st->s.lambda_c = lambda_c;
	st->s.cutoff_frequency_high = cutoff_high;
	st->s.cutoff_frequency_low = cutoff_low;
	st->s.chrom_attenuation = chrom_attenuation;
	st->s.exaggeration_factor = exaggeration;
}

EVM_API void evm_reset(void *handle)
{
	auto *st = static_cast<evm_state *>(handle);
	st->frame_num = 0;
	st->low_pass1.clear();
	st->low_pass2.clear();
	st->filtered.clear();
}

EVM_API void evm_destroy(void *handle) { delete static_cast<evm_state *>(handle); }

// bgr_in / bgr_out: h*w*3 uint8, row-major, contiguous. Returns frames processed.
EVM_API int evm_process(void *handle, const unsigned char *bgr_in, int w, int h, unsigned char *bgr_out)
{
	auto *st = static_cast<evm_state *>(handle);
	auto &settings = st->s;
	if (w != st->w || h != st->h)
	{
		evm_reset(handle);
		st->w = w;
		st->h = h;
	}

	cv::Mat frame(h, w, CV_8UC3, const_cast<unsigned char *>(bgr_in));

	// ---- input stage (was the input thread) ----
	auto input = frame.clone();

	// Convert to LAB color space
	input.convertTo(input, CV_32FC3, 1.0 / 255.0f);
	cv::cvtColor(input, input, cv::COLOR_BGR2Lab);

	// Pyramid decomposition
	std::vector<cv::Mat> pyramid;
	{
		auto current = input;
		for (int l = 0; l < settings.levels; l++)
		{
			cv::Mat down, up;

			pyrDown(current, down);
			pyrUp(down, up, current.size());

			pyramid.push_back(current - up);
			current = down;
		}

		pyramid.push_back(current);
		pyramid.push_back(input);
	}

	// ---- processing stage (was the main thread) ----
	auto &low_pass1 = st->low_pass1;
	auto &low_pass2 = st->low_pass2;
	auto &filtered = st->filtered;
	auto &frame_num = st->frame_num;

	if (frame_num == 0)
	{
		for (int l = 0; l < (int)pyramid.size(); l++)
		{
			filtered.push_back(pyramid[l].clone());
			low_pass1.push_back(pyramid[l].clone());
			low_pass2.push_back(pyramid[l].clone());
		}
	}

	if (frame_num > 0)
	{
		auto delta = settings.lambda_c / 8.0 / (1.0 + settings.alpha);
		auto lambda = sqrt((float)(w * w + h * h)) / 3;

		// Process levels of pyramid
		std::vector<std::thread> workers;
		for (int level = settings.levels; level >= 0; level--)
		{
			workers.push_back(std::thread([&, level, lambda]() {
				// First or last level we mostly ignore
				if (level == settings.levels || level == 0)
				{
					filtered[level] *= 0;
					return;
				}

				// Temporal IIR Filter
				low_pass1[level] = (1 - settings.cutoff_frequency_high) * low_pass1[level] + settings.cutoff_frequency_high * pyramid[level];
				low_pass2[level] = (1 - settings.cutoff_frequency_low) * low_pass2[level] + settings.cutoff_frequency_low * pyramid[level];
				filtered[level] = low_pass1[level] - low_pass2[level];

				// Amplify
				auto current_alpha = (lambda / delta / 8 - 1) * settings.exaggeration_factor;
				filtered[level] *= std::min(settings.alpha, current_alpha);
			}));

			lambda /= 2.0;
		}

		std::for_each(workers.begin(), workers.end(), [](std::thread &t) {
			t.join();
		});
	}

	// Pyramid reconstruction
	cv::Mat motion;
	{
		auto current = filtered[settings.levels];
		for (int level = settings.levels - 1; level >= 0; --level)
		{
			cv::Mat up;
			pyrUp(current, up, filtered[level].size());
			current = up + filtered[level];
		}
		motion = current;
	}

	cv::Mat in_lab = pyramid[settings.levels + 1];
	if (frame_num > 0)
	{
		// Attenuate I, Q channels
		cv::Mat planes[3];
		split(motion, planes);
		planes[1] = planes[1] * settings.chrom_attenuation;
		planes[2] = planes[2] * settings.chrom_attenuation;
		cv::merge(planes, 3, motion);

		// Combine
		motion = in_lab + motion;
	}
	else
	{
		motion = in_lab.clone();
	}
	frame_num++;

	// ---- output stage (was the output thread) ----
	cv::Mat amplified = motion;
	cv::cvtColor(amplified, amplified, cv::COLOR_Lab2BGR);
	amplified.convertTo(amplified, CV_8UC3, 255.0, 1.0 / 255.0);

	cv::Mat out(h, w, CV_8UC3, bgr_out);
	amplified.copyTo(out);
	return frame_num;
}
