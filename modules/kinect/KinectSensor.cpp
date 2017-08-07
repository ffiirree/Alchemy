#include"KinectSensor.h"

#ifdef USING_KINECT
#include <iostream>
#include <malloc.h>
#include "zcore\debug.h"

using namespace std;

KinectSensor::KinectSensor()
{
	pSensor = nullptr;
	pDepthFrameReader = nullptr;
	pColorFrameReader = nullptr;
	pInfraFrameReader = nullptr;

	depthMinReliableDistance = 0;
	depthMaxReliableDistance = 0;

	colorImg = cv::Mat::zeros(colorImgHeight, colorImgWidth, CV_8UC4);
	depthImg = cv::Mat::zeros(depthImgHeight, depthImgWidth, CV_16UC1);
	infraImg = cv::Mat::zeros(infraImgHeight, infraImgWidth, CV_16UC1);

	KinectInitialize(FrameTypes_All);
}

KinectSensor::KinectSensor(FrameTypes _type)
{
	pSensor = nullptr;
	pDepthFrameReader = nullptr;
	pColorFrameReader = nullptr;
	pInfraFrameReader = nullptr;

	depthMinReliableDistance = 0;
	depthMaxReliableDistance = 0;

	if (_type == FrameTypes_Color) {
		colorImg = cv::Mat::zeros(colorImgHeight, colorImgWidth, CV_8UC4);
	}
	else if (_type == FrameTypes_Depth) {
		depthImg = cv::Mat::zeros(depthImgHeight, depthImgWidth, CV_16UC1);
	}
	else if (_type == FrameTypes_Infra) {
		infraImg = cv::Mat::zeros(infraImgHeight, infraImgWidth, CV_16UC1);
	}
	else if (_type == FrameTypes_All) {
		colorImg = cv::Mat::zeros(colorImgHeight, colorImgWidth, CV_8UC4);
		depthImg = cv::Mat::zeros(depthImgHeight, depthImgWidth, CV_16UC1);
		infraImg = cv::Mat::zeros(infraImgHeight, infraImgWidth, CV_16UC1);
	}
	KinectInitialize(_type);
}

/**
 * @brief 初始化Kinect传感器，分别获取source并且打开Reader
 *		Kinect 获取数据的流程大体遵循：Source->Reader->Frame流程来获取最终的数据
 * @attention 注意获取资源后的释放
 */
HRESULT KinectSensor::KinectInitialize(FrameTypes _type)
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&pSensor);
	if (FAILED(hr)) {
		_log_("Sensor initialize filed!");
		return hr;
	}

	if (pSensor) {
		hr = pSensor->Open();

		if (_type == FrameTypes_Depth) {
			IDepthFrameSource * pDepthFrameSource = nullptr;

			if (SUCCEEDED(hr)) {
				hr = pSensor->get_DepthFrameSource(&pDepthFrameSource);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrameSource->OpenReader(&pDepthFrameReader);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrameReader->SubscribeFrameArrived(&depthFrameEvent);
			}
			SafeRelease(&pDepthFrameSource);
		}
		else if (_type == FrameTypes_Color) {
			IColorFrameSource * pColorFrameSource = nullptr;
			if (SUCCEEDED(hr)) {
				hr = pSensor->get_ColorFrameSource(&pColorFrameSource);
			}
			if (SUCCEEDED(hr)) {
				hr = pColorFrameSource->OpenReader(&pColorFrameReader);
			}
			if (SUCCEEDED(hr)) {
				hr = pColorFrameReader->SubscribeFrameArrived(&colorFrameEvent);
			}
			SafeRelease(&pColorFrameSource);
		}
		else if (_type == FrameTypes_Infra) {
			IInfraredFrameSource * pInfraFrameSource = nullptr;
			if (SUCCEEDED(hr)) {
				hr = pSensor->get_InfraredFrameSource(&pInfraFrameSource);
			}
			if (SUCCEEDED(hr)) {
				hr = pInfraFrameSource->OpenReader(&pInfraFrameReader);
			}
			if (SUCCEEDED(hr)) {
				hr = pInfraFrameReader->SubscribeFrameArrived(&infraFrameEvent);
			}
			SafeRelease(&pInfraFrameSource);
		}
		else if (_type == FrameTypes_All) {
			if (SUCCEEDED(hr)) {
				hr = pSensor->OpenMultiSourceFrameReader(FrameSourceTypes_Depth
					| FrameSourceTypes_Infrared | FrameSourceTypes_Color,
					&pMultiSourceFrameReader);
			}
			if (SUCCEEDED(hr)) {
				hr = pMultiSourceFrameReader->SubscribeMultiSourceFrameArrived(&allFramesEvent);
			}
		}
	} // !if (pSensor)

	if (!pSensor | FAILED(hr)) {
		_log_("No device ready!");
		return E_FAIL;
	}
	return hr;
}


/**
 * @brief 更新图形数据
 * @attention AcquireLatestFrame()函数不一定会成功返回数据，一定要做好处理，
 *            特别是使用轮询的时候，经常失败；使用event后基本都会成功
 */
HRESULT KinectSensor::update(FrameTypes _type)
{
	HRESULT hr;
	if (_type == FrameTypes_Depth) {
		hr = updateDepthData();
		if (FAILED(hr)) {
			//_log_("Update depth data error!!\n");
			return hr;
		}
	}
	else if (_type == FrameTypes_Color) {
		hr = updateColorData();
		if (FAILED(hr)) {
			//_log_("Update color data error!!\n");
			return hr;
		}
	}
	else if(_type == FrameTypes_Infra) {
		hr = updateInfraData();
		if (FAILED(hr)) {
			//_log_("Update IR data error!!\n");
			return hr;
		}
	}
	else if (_type == FrameTypes_All) {
		hr = updateAll();
		if (FAILED(hr)) {
			return hr;
		}
	}
	return S_OK;
}

HRESULT KinectSensor::updateAll()
{
	if (!pMultiSourceFrameReader)
		return S_FALSE;

	IMultiSourceFrame *pMultiSourceFrame = nullptr;
	IDepthFrame* pDepthFrame = nullptr;
	IColorFrame* pColorFrame = nullptr;
	IInfraredFrame* pInfraFrame = nullptr;
	HRESULT hr = pMultiSourceFrameReader->AcquireLatestFrame(&pMultiSourceFrame);

	if (SUCCEEDED(hr)) {
		IColorFrameReference *pColorFrameReference = nullptr;
		IDepthFrameReference * pDepthFrameReference = nullptr;
		IInfraredFrameReference *pInfraFrameReference = nullptr;

		// color 
		HRESULT hrColor = pMultiSourceFrame->get_ColorFrameReference(&pColorFrameReference);
		if (SUCCEEDED(hrColor)) {
			hrColor = pColorFrameReference->AcquireFrame(&pColorFrame);
		}
		SafeRelease(&pColorFrameReference);

		// depth
		HRESULT hrDepth = pMultiSourceFrame->get_DepthFrameReference(&pDepthFrameReference);
		if (SUCCEEDED(hrDepth)) {
			hrDepth = pDepthFrameReference->AcquireFrame(&pDepthFrame);
		}
		SafeRelease(&pDepthFrameReference);

		// infra
		HRESULT hrInfra = pMultiSourceFrame->get_InfraredFrameReference(&pInfraFrameReference);
		if (SUCCEEDED(hrInfra)) {
			hrInfra = pInfraFrameReference->AcquireFrame(&pInfraFrame);
		}
		SafeRelease(&pInfraFrameReference);


		// Cheak all data-------------------------
		if (SUCCEEDED(hrColor) && SUCCEEDED(hrDepth) && SUCCEEDED(hrInfra)) {
			// depth data
			// 获取深度图像的时间戳
			hr = pDepthFrame->get_RelativeTime(&depthFrameTimestamp);

			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->get_DepthMinReliableDistance(&depthMinReliableDistance);
			}
			if (SUCCEEDED(hr)) {
				hr = pDepthFrame->get_DepthMaxReliableDistance(&depthMaxReliableDistance);
			}
			else {
				_log_("get_DepthMaxReliableDistance faild!\n");
			}
			if (SUCCEEDED(hr)) {
				pDepthFrame->CopyFrameDataToArray(512 * 424, reinterpret_cast<UINT16*>(depthImg.data));
			}
			else {
				_log_("AccessUnderlyingBuffer faild!\n");
			}
			

			// color data-------------------------
			ColorImageFormat imgFmt = ColorImageFormat_None;
			hr = pColorFrame->get_RelativeTime(&colorFrameTimestamp);
			if (SUCCEEDED(hr)) {
				hr = pColorFrame->get_RawColorImageFormat(&imgFmt);
			}

			if (SUCCEEDED(hr)) {
				if (imgFmt == ColorImageFormat_Bgra) {
					hr = pColorFrame->AccessRawUnderlyingBuffer(&colorBufferSize, reinterpret_cast<BYTE**>(&colorImg.data));
				}
				else if (colorImg.data) {
					colorBufferSize = colorImgHeight * colorImgWidth * sizeof(RGBQUAD);
					hr = pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(colorImg.data), ColorImageFormat_Bgra);
				}
				else {
					hr = E_FAIL;
				}
			}

			// infra data--------------------------
			hr = pInfraFrame->get_RelativeTime(&infraFrameTimestamp);

			if (SUCCEEDED(hr)) {
				hr = pInfraFrame->CopyFrameDataToArray(424 * 512, reinterpret_cast<UINT16*>(infraImg.data));
			}
		} // ! if (SUCCEEDED(hrColor) && SUCCEEDED(hrDepth)) 

		SafeRelease(&pDepthFrame);
		SafeRelease(&pColorFrame);
		SafeRelease(&pInfraFrame);
		SafeRelease(&pMultiSourceFrame);
	}
	return hr;
}


/**
 * @attention AccessUnderlyingBuffer()函数获取的只是指向数据的指针
 *    可以使用CopyFrameDataToArray()将数据拷贝出来
 */
HRESULT KinectSensor::updateDepthData()
{
	if (!pDepthFrameReader)
		return S_FALSE;

	IDepthFrame* pDepthFrame = nullptr;
	HRESULT hr = pDepthFrameReader->AcquireLatestFrame(&pDepthFrame);

	if (SUCCEEDED(hr)) {
		// 获取深度图像的时间戳
		hr = pDepthFrame->get_RelativeTime(&depthFrameTimestamp);

		if (SUCCEEDED(hr)) {
			hr = pDepthFrame->get_DepthMinReliableDistance(&depthMinReliableDistance);
		}
		if (SUCCEEDED(hr)) {
			hr = pDepthFrame->get_DepthMaxReliableDistance(&depthMaxReliableDistance);
		}
		else {
			_log_("get_DepthMaxReliableDistance faild!\n");
		}
		if (SUCCEEDED(hr)) {
			pDepthFrame->CopyFrameDataToArray(512 * 424, reinterpret_cast<UINT16*>(depthImg.data));
		}
		else {
			_log_("AccessUnderlyingBuffer faild!\n");
		}
	}
	else {
		_log_("AcquireLatestFrame faild!\n");
	}
	SafeRelease(&pDepthFrame);
	return hr;
}

HRESULT KinectSensor::updateColorData()
{
	if (!pColorFrameReader)
		return S_FALSE;

	IColorFrame * pColorFrame = nullptr;
	HRESULT hr = pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	ColorImageFormat imgFmt = ColorImageFormat_None;
	if (SUCCEEDED(hr)) {
		hr = pColorFrame->get_RelativeTime(&colorFrameTimestamp);
		if (SUCCEEDED(hr)) {
			hr = pColorFrame->get_RawColorImageFormat(&imgFmt);
		}

		if (SUCCEEDED(hr)) {
			if (imgFmt == ColorImageFormat_Bgra) {
				hr = pColorFrame->AccessRawUnderlyingBuffer(&colorBufferSize, reinterpret_cast<BYTE**>(&colorImg.data));
			}
			else if (colorImg.data) {
				colorBufferSize = colorImgHeight * colorImgWidth * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(colorBufferSize, reinterpret_cast<BYTE*>(colorImg.data), ColorImageFormat_Bgra);
			}
			else {
				hr = E_FAIL;
			}
		}
		else {
			_log_("get_RawColorImageFormat faild!\n");
		}
		SafeRelease(&pColorFrame);
	}
	else {
		_log_("AcquireLatestFrame faild!\n");
	}
	return hr;
}

HRESULT KinectSensor::updateInfraData()
{
	IInfraredFrame * pInfraFrame = nullptr;
	HRESULT hr = pInfraFrameReader->AcquireLatestFrame(&pInfraFrame);

	if (SUCCEEDED(hr)) {
		hr = pInfraFrame->get_RelativeTime(&infraFrameTimestamp);

		if (SUCCEEDED(hr)) {
			hr = pInfraFrame->CopyFrameDataToArray(424 * 512, reinterpret_cast<UINT16*>(infraImg.data));
		}
		SafeRelease(&pInfraFrame);
	}

	return hr;
}


template< class T >
inline void KinectSensor::SafeRelease(T** ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT = nullptr;
	}
}

// Safe release for interfaces
template<class Interface>
inline void KinectSensor::SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != nullptr)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = nullptr;
	}
}


KinectSensor::~KinectSensor()
{
	if (pDepthFrameReader) {
		pDepthFrameReader->UnsubscribeFrameArrived(depthFrameEvent);
		depthFrameEvent = NULL;

		pDepthFrameReader->Release();
		pDepthFrameReader = nullptr;
	}
	
	if (pColorFrameReader) {
		pColorFrameReader->UnsubscribeFrameArrived(colorFrameEvent);
		colorFrameEvent = NULL;

		pColorFrameReader->Release();
		pColorFrameReader = nullptr;
	}
	
	if (pInfraFrameReader) {
		pInfraFrameReader->UnsubscribeFrameArrived(infraFrameEvent);
		CloseHandle((HANDLE)infraFrameEvent);
		infraFrameEvent = NULL;

		pInfraFrameReader->Release();
		pInfraFrameReader = nullptr;
	}
	
	if (pMultiSourceFrameReader) {
		pMultiSourceFrameReader->UnsubscribeMultiSourceFrameArrived(allFramesEvent);
		allFramesEvent = NULL;

		pMultiSourceFrameReader->Release();
		pMultiSourceFrameReader = nullptr;
	}
	

	if (pSensor)
	{
		pSensor->Close();
		SafeRelease(&pSensor);
		pSensor = nullptr;
	}
}

HRESULT KinectSensor::isNewFrameArrived(FrameTypes frameTypes)
{
	if (frameTypes == FrameTypes_Depth) {
		if (depthFrameEvent != NULL 
			&& WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)depthFrameEvent, 0)) {
			ResetEvent(HANDLE(depthFrameEvent));
			return S_OK;
		}
		else {
			return S_FALSE;
		}
	}
	else if (frameTypes == FrameTypes_Infra) {
		if (infraFrameEvent != NULL 
			&& WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)infraFrameEvent, 0)) {
			ResetEvent(HANDLE(infraFrameEvent));
			return S_OK;
		}
		else {
			return S_FALSE;
		}
	}
	else if (frameTypes == FrameTypes_Color) {
		if (colorFrameEvent != NULL 
			&& WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)colorFrameEvent, 0)) {
			ResetEvent(HANDLE(colorFrameEvent));
			return S_OK;
		}
		else {
			return S_FALSE;
		}
	}
	else if (frameTypes == FrameTypes_All) {
		if (allFramesEvent != NULL
			&& WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)allFramesEvent, 0)) {
			ResetEvent(HANDLE(allFramesEvent));
			return S_OK;
		}
		else {
			return S_FALSE;
		}
	}
	else {
		return S_FALSE;
	}
}
#endif // USING_KINECT