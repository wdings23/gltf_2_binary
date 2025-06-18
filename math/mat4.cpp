#include <stdint.h>
#include <math.h>

#ifdef _MSC_VER
#include <corecrt_math_defines.h>
#endif // MSC_VER

#include "mat4.h"
#include "quaternion.h"

#include <float.h>

/*
**
*/
mat4 invert(mat4 const& m)
{
    float inv[16], invOut[16], det;
    int i;
    
    inv[0] = m.mafEntries[5]  * m.mafEntries[10] * m.mafEntries[15] -
    m.mafEntries[5]  * m.mafEntries[11] * m.mafEntries[14] -
    m.mafEntries[9]  * m.mafEntries[6]  * m.mafEntries[15] +
    m.mafEntries[9]  * m.mafEntries[7]  * m.mafEntries[14] +
    m.mafEntries[13] * m.mafEntries[6]  * m.mafEntries[11] -
    m.mafEntries[13] * m.mafEntries[7]  * m.mafEntries[10];
    
    inv[4] = -m.mafEntries[4]  * m.mafEntries[10] * m.mafEntries[15] +
    m.mafEntries[4]  * m.mafEntries[11] * m.mafEntries[14] +
    m.mafEntries[8]  * m.mafEntries[6]  * m.mafEntries[15] -
    m.mafEntries[8]  * m.mafEntries[7]  * m.mafEntries[14] -
    m.mafEntries[12] * m.mafEntries[6]  * m.mafEntries[11] +
    m.mafEntries[12] * m.mafEntries[7]  * m.mafEntries[10];
    
    inv[8] = m.mafEntries[4]  * m.mafEntries[9] * m.mafEntries[15] -
    m.mafEntries[4]  * m.mafEntries[11] * m.mafEntries[13] -
    m.mafEntries[8]  * m.mafEntries[5] * m.mafEntries[15] +
    m.mafEntries[8]  * m.mafEntries[7] * m.mafEntries[13] +
    m.mafEntries[12] * m.mafEntries[5] * m.mafEntries[11] -
    m.mafEntries[12] * m.mafEntries[7] * m.mafEntries[9];
    
    inv[12] = -m.mafEntries[4]  * m.mafEntries[9] * m.mafEntries[14] +
    m.mafEntries[4]  * m.mafEntries[10] * m.mafEntries[13] +
    m.mafEntries[8]  * m.mafEntries[5] * m.mafEntries[14] -
    m.mafEntries[8]  * m.mafEntries[6] * m.mafEntries[13] -
    m.mafEntries[12] * m.mafEntries[5] * m.mafEntries[10] +
    m.mafEntries[12] * m.mafEntries[6] * m.mafEntries[9];
    
    inv[1] = -m.mafEntries[1]  * m.mafEntries[10] * m.mafEntries[15] +
    m.mafEntries[1]  * m.mafEntries[11] * m.mafEntries[14] +
    m.mafEntries[9]  * m.mafEntries[2] * m.mafEntries[15] -
    m.mafEntries[9]  * m.mafEntries[3] * m.mafEntries[14] -
    m.mafEntries[13] * m.mafEntries[2] * m.mafEntries[11] +
    m.mafEntries[13] * m.mafEntries[3] * m.mafEntries[10];
    
    inv[5] = m.mafEntries[0]  * m.mafEntries[10] * m.mafEntries[15] -
    m.mafEntries[0]  * m.mafEntries[11] * m.mafEntries[14] -
    m.mafEntries[8]  * m.mafEntries[2] * m.mafEntries[15] +
    m.mafEntries[8]  * m.mafEntries[3] * m.mafEntries[14] +
    m.mafEntries[12] * m.mafEntries[2] * m.mafEntries[11] -
    m.mafEntries[12] * m.mafEntries[3] * m.mafEntries[10];
    
    inv[9] = -m.mafEntries[0]  * m.mafEntries[9] * m.mafEntries[15] +
    m.mafEntries[0]  * m.mafEntries[11] * m.mafEntries[13] +
    m.mafEntries[8]  * m.mafEntries[1] * m.mafEntries[15] -
    m.mafEntries[8]  * m.mafEntries[3] * m.mafEntries[13] -
    m.mafEntries[12] * m.mafEntries[1] * m.mafEntries[11] +
    m.mafEntries[12] * m.mafEntries[3] * m.mafEntries[9];
    
    inv[13] = m.mafEntries[0]  * m.mafEntries[9] * m.mafEntries[14] -
    m.mafEntries[0]  * m.mafEntries[10] * m.mafEntries[13] -
    m.mafEntries[8]  * m.mafEntries[1] * m.mafEntries[14] +
    m.mafEntries[8]  * m.mafEntries[2] * m.mafEntries[13] +
    m.mafEntries[12] * m.mafEntries[1] * m.mafEntries[10] -
    m.mafEntries[12] * m.mafEntries[2] * m.mafEntries[9];
    
    inv[2] = m.mafEntries[1]  * m.mafEntries[6] * m.mafEntries[15] -
    m.mafEntries[1]  * m.mafEntries[7] * m.mafEntries[14] -
    m.mafEntries[5]  * m.mafEntries[2] * m.mafEntries[15] +
    m.mafEntries[5]  * m.mafEntries[3] * m.mafEntries[14] +
    m.mafEntries[13] * m.mafEntries[2] * m.mafEntries[7] -
    m.mafEntries[13] * m.mafEntries[3] * m.mafEntries[6];
    
    inv[6] = -m.mafEntries[0]  * m.mafEntries[6] * m.mafEntries[15] +
    m.mafEntries[0]  * m.mafEntries[7] * m.mafEntries[14] +
    m.mafEntries[4]  * m.mafEntries[2] * m.mafEntries[15] -
    m.mafEntries[4]  * m.mafEntries[3] * m.mafEntries[14] -
    m.mafEntries[12] * m.mafEntries[2] * m.mafEntries[7] +
    m.mafEntries[12] * m.mafEntries[3] * m.mafEntries[6];
    
    inv[10] = m.mafEntries[0]  * m.mafEntries[5] * m.mafEntries[15] -
    m.mafEntries[0]  * m.mafEntries[7] * m.mafEntries[13] -
    m.mafEntries[4]  * m.mafEntries[1] * m.mafEntries[15] +
    m.mafEntries[4]  * m.mafEntries[3] * m.mafEntries[13] +
    m.mafEntries[12] * m.mafEntries[1] * m.mafEntries[7] -
    m.mafEntries[12] * m.mafEntries[3] * m.mafEntries[5];
    
    inv[14] = -m.mafEntries[0]  * m.mafEntries[5] * m.mafEntries[14] +
    m.mafEntries[0]  * m.mafEntries[6] * m.mafEntries[13] +
    m.mafEntries[4]  * m.mafEntries[1] * m.mafEntries[14] -
    m.mafEntries[4]  * m.mafEntries[2] * m.mafEntries[13] -
    m.mafEntries[12] * m.mafEntries[1] * m.mafEntries[6] +
    m.mafEntries[12] * m.mafEntries[2] * m.mafEntries[5];
    
    inv[3] = -m.mafEntries[1] * m.mafEntries[6] * m.mafEntries[11] +
    m.mafEntries[1] * m.mafEntries[7] * m.mafEntries[10] +
    m.mafEntries[5] * m.mafEntries[2] * m.mafEntries[11] -
    m.mafEntries[5] * m.mafEntries[3] * m.mafEntries[10] -
    m.mafEntries[9] * m.mafEntries[2] * m.mafEntries[7] +
    m.mafEntries[9] * m.mafEntries[3] * m.mafEntries[6];
    
    inv[7] = m.mafEntries[0] * m.mafEntries[6] * m.mafEntries[11] -
    m.mafEntries[0] * m.mafEntries[7] * m.mafEntries[10] -
    m.mafEntries[4] * m.mafEntries[2] * m.mafEntries[11] +
    m.mafEntries[4] * m.mafEntries[3] * m.mafEntries[10] +
    m.mafEntries[8] * m.mafEntries[2] * m.mafEntries[7] -
    m.mafEntries[8] * m.mafEntries[3] * m.mafEntries[6];
    
    inv[11] = -m.mafEntries[0] * m.mafEntries[5] * m.mafEntries[11] +
    m.mafEntries[0] * m.mafEntries[7] * m.mafEntries[9] +
    m.mafEntries[4] * m.mafEntries[1] * m.mafEntries[11] -
    m.mafEntries[4] * m.mafEntries[3] * m.mafEntries[9] -
    m.mafEntries[8] * m.mafEntries[1] * m.mafEntries[7] +
    m.mafEntries[8] * m.mafEntries[3] * m.mafEntries[5];
    
    inv[15] = m.mafEntries[0] * m.mafEntries[5] * m.mafEntries[10] -
    m.mafEntries[0] * m.mafEntries[6] * m.mafEntries[9] -
    m.mafEntries[4] * m.mafEntries[1] * m.mafEntries[10] +
    m.mafEntries[4] * m.mafEntries[2] * m.mafEntries[9] +
    m.mafEntries[8] * m.mafEntries[1] * m.mafEntries[6] -
    m.mafEntries[8] * m.mafEntries[2] * m.mafEntries[5];
    
    det = m.mafEntries[0] * inv[0] + m.mafEntries[1] * inv[4] + m.mafEntries[2] * inv[8] + m.mafEntries[3] * inv[12];
    if(det <= 1.0e-5)
    {
        for(i = 0; i < 16; i++)
            invOut[i] = FLT_MAX;
    }
    else
    {
        det = 1.0f / det;

        for(i = 0; i < 16; i++)
            invOut[i] = inv[i] * det;
    }
    
    return mat4(invOut);
}

/*
**
*/
void mul(mat4* pResult, mat4 const& m0, mat4 const& m1)
{
    for(uint32_t i = 0; i < 4; i++)
    {
        for(uint32_t j = 0; j < 4; j++)
        {
            float fResult = 0.0f;
            for(uint32_t k = 0; k < 4; k++)
            {
                uint32_t iIndex0 = (i << 2) + k;
                uint32_t iIndex1 = (k << 2) + j;
                fResult += (m0.mafEntries[iIndex0] * m1.mafEntries[iIndex1]);
            }
            
            pResult->mafEntries[(i << 2) + j] = fResult;
        }
    }
}

/*
**
*/
void mul(mat4& result, mat4 const& m0, mat4 const& m1)
{
    for(uint32_t i = 0; i < 4; i++)
    {
        for(uint32_t j = 0; j < 4; j++)
        {
            float fResult = 0.0f;
            for(uint32_t k = 0; k < 4; k++)
            {
                uint32_t iIndex0 = (i << 2) + k;
                uint32_t iIndex1 = (k << 2) + j;
                fResult += (m0.mafEntries[iIndex0] * m1.mafEntries[iIndex1]);
            }

            result.mafEntries[(i << 2) + j] = fResult;
        }
    }
}

/*
**
*/
vec4 mul(float4 const& v, mat4 const& m)
{
    return m * v;
}

/*
**
*/
mat4 translate(float fX, float fY, float fZ)
{
    float afVal[16] =
    {
        1.0f, 0.0f, 0.0f, fX,
        0.0f, 1.0f, 0.0f, fY,
        0.0f, 0.0f, 1.0f, fZ,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 translate(vec4 const& position)
{
    float afVal[16] =
    {
        1.0f, 0.0f, 0.0f, position.x,
        0.0f, 1.0f, 0.0f, position.y,
        0.0f, 0.0f, 1.0f, position.z,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 transpose(mat4 const& m)
{
    float afVal[16] =
    {
        m.mafEntries[0], m.mafEntries[4], m.mafEntries[8], m.mafEntries[12],
        m.mafEntries[1], m.mafEntries[5], m.mafEntries[9], m.mafEntries[13],
        m.mafEntries[2], m.mafEntries[6], m.mafEntries[10], m.mafEntries[14],
        m.mafEntries[3], m.mafEntries[7], m.mafEntries[11], m.mafEntries[15],
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 mat4::operator * (mat4 const& m) const
{
    float afResults[16];
    
    for(uint32_t i = 0; i < 4; i++)
    {
        for(uint32_t j = 0; j < 4; j++)
        {
            uint32_t iIndex = (i << 2) + j;
            afResults[iIndex] = 0.0f;
            for(uint32_t k = 0; k < 4; k++)
            {
                uint32_t iIndex0 = (i << 2) + k;
                uint32_t iIndex1 = (k << 2) + j;
                afResults[iIndex] += (mafEntries[iIndex0] * m.mafEntries[iIndex1]);
            }
        }
    }
    
    return mat4(afResults);
}

/*
**
*/
mat4 mat4::operator + (mat4 const& m) const
{
    float afResults[16];

    for(uint32_t i = 0; i < 16; i++)
    {
        afResults[i] = mafEntries[i] + m.mafEntries[i];
    }

    return mat4(afResults);
}

/*
**
*/
void mat4::operator += (mat4 const& m)
{
    for(uint32_t i = 0; i < 16; i++)
    {
        mafEntries[i] += m.mafEntries[i];
    }
}

/*
**
*/
mat4 perspectiveProjection(float fFOV,
                           uint32_t iWidth,
                           uint32_t iHeight,
                           float fFar,
                           float fNear)
{
    float fFD = 1.0f / tanf(fFOV * 0.5f);
    float fAspect = (float)iWidth / (float)iHeight;
    float fOneOverAspect = 1.0f / fAspect;
    float fOneOverFarMinusNear = 1.0f / (fFar - fNear);
    
    float afVal[16];
    memset(afVal, 0, sizeof(afVal));
    afVal[0] = fFD * fOneOverAspect;
    afVal[5] = -fFD;
    afVal[10] = -(fFar + fNear) * fOneOverFarMinusNear;
    afVal[14] = -1.0f;
    afVal[11] = -2.0f * fFar * fNear * fOneOverFarMinusNear;
    afVal[15] = 0.0f;
    
#if defined(__APPLE__) || defined(TARGET_IOS)
    afVal[10] = -fFar * fOneOverFarMinusNear;
    afVal[11] = -fFar * fNear * fOneOverFarMinusNear;
#else
#if !defined(GLES_RENDER)
	afVal[5] *= -1.0f;
#endif // GLES_RENDER

#endif // __APPLE__
    
    return mat4(afVal);
}

/*
**
*/
mat4 perspectiveProjectionNegOnePosOne(
    float fFOV,
    uint32_t iWidth,
    uint32_t iHeight,
    float fFar,
    float fNear)
{
    float fFD = 1.0f / tanf(fFOV * 0.5f);
    float fAspect = (float)iWidth / (float)iHeight;
    float fOneOverAspect = 1.0f / fAspect;
    float fOneOverFarMinusNear = 1.0f / (fFar - fNear);

    float afVal[16];
    memset(afVal, 0, sizeof(afVal));
    afVal[0] = fFD * fOneOverAspect;
    afVal[5] = fFD;
    afVal[10] = -(fFar + fNear) * fOneOverFarMinusNear;
    afVal[11] = -1.0f;
    afVal[14] = -2.0f * fFar * fNear * fOneOverFarMinusNear;
    afVal[15] = 0.0f;

    return mat4(afVal);
}

/*
**
*/
mat4 perspectiveProjection2(
    float fFOV,
    uint32_t iWidth,
    uint32_t iHeight,
    float fFar,
    float fNear)
{
    float fFD = 1.0f / tanf(fFOV * 0.5f);
    float fAspect = (float)iWidth / (float)iHeight;
    float fOneOverAspect = 1.0f / fAspect;
    float fOneOverFarMinusNear = 1.0f / (fFar - fNear);

    float afVal[16];
    memset(afVal, 0, sizeof(afVal));
    afVal[0] = -fFD * fOneOverAspect;
    afVal[5] = fFD;
    afVal[10] = -fFar * fOneOverFarMinusNear;
    afVal[11] = fFar * fNear * fOneOverFarMinusNear;
    afVal[14] = -1.0f;
    afVal[15] = 0.0f;

    return mat4(afVal);
}

/*
**
*/
mat4 orthographicProjection(float fLeft,
                            float fRight,
                            float fTop,
                            float fBottom,
                            float fFar,
                            float fNear,
                            bool bInvertY)
{
    float fWidth = fRight - fLeft;
    float fHeight = fTop - fBottom;
    
    float fFarMinusNear = fFar - fNear;
    
    float afVal[16];
    memset(afVal, 0, sizeof(afVal));
    
    afVal[0] = -2.0f / fWidth;
    afVal[3] = -(fRight + fLeft) / (fRight - fLeft);
    afVal[5] = 2.0f / fHeight;
    afVal[7] = -(fTop + fBottom) / (fTop - fBottom);

    afVal[10] = 1.0f / fFarMinusNear;
    afVal[11] = -fNear / fFarMinusNear;

    afVal[15] = 1.0f;
    
    if(bInvertY)
    {
        afVal[5] = -afVal[5];
    }
    
    afVal[0] = -afVal[0];
    
    return mat4(afVal);
}

/*
**
*/
mat4 makeViewMatrix(vec3 const& eyePos, vec3 const& lookAt, vec3 const& up)
{
    vec3 dir = lookAt - eyePos;
    dir = normalize(dir);
    
    vec3 tangent = normalize(cross(up, dir));
    vec3 binormal = normalize(cross(dir, tangent));
    
    float afValue[16] =
    {
        tangent.x, tangent.y, tangent.z, 0.0f,
        binormal.x, binormal.y, binormal.z, 0.0f,
        dir.x, dir.y, dir.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    
    mat4 xform(afValue);
    
    mat4 translation;
    translation.mafEntries[3] = eyePos.x;
    translation.mafEntries[7] = eyePos.y;
    translation.mafEntries[11] = eyePos.z;
    
    return (xform * translation);
}

/*
**
*/
mat4 makeViewMatrix2(vec3 const& eyePos, vec3 const& lookAt, vec3 const& up)
{
    vec3 dir = lookAt - eyePos;
    dir = normalize(dir);

    vec3 tangent = normalize(cross(up, dir));
    vec3 binormal = normalize(cross(dir, tangent));

    float afValue[16] =
    {
        tangent.x, tangent.y, tangent.z, 0.0f,
        binormal.x, binormal.y, binormal.z, 0.0f,
        dir.x, dir.y, dir.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };

    mat4 xform(afValue);

    mat4 translation;
    translation.mafEntries[3] = -eyePos.x;
    translation.mafEntries[7] = -eyePos.y;
    translation.mafEntries[11] = -eyePos.z;

    //return (translation * xform);
    return (xform * translation);
}

/*
**
*/
mat4 rotateMatrixX(float fAngle)
{
    float afVal[16] =
    {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, cosf(fAngle), -sinf(fAngle), 0.0f,
        0.0f, sinf(fAngle), cosf(fAngle), 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 rotateMatrixY(float fAngle)
{
    float afVal[16] =
    {
        cosf(fAngle), 0.0f, sinf(fAngle), 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        -sinf(fAngle), 0.0f, cosf(fAngle), 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 rotateMatrixZ(float fAngle)
{
    float afVal[16] =
    {
        cosf(fAngle), -sinf(fAngle), 0.0f, 0.0f,
        sinf(fAngle), cosf(fAngle), 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 scale(float fX, float fY, float fZ)
{
    float afVal[16] =
    {
        fX, 0.0f, 0.0f, 0.0f,
        0.0f, fY, 0.0f, 0.0f,
        0.0f, 0.0f, fZ, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
mat4 scale(vec4 const& scale)
{
    float afVal[16] =
    {
        scale.x, 0.0f, 0.0f, 0.0f,
        0.0f, scale.y, 0.0f, 0.0f,
        0.0f, 0.0f, scale.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    
    return mat4(afVal);
}

/*
**
*/
bool mat4::operator == (mat4 const& m) const
{
    bool bRet = true;
    for(uint32_t i = 0; i < 16; i++)
    {
        float fDiff = fabsf(m.mafEntries[i] - mafEntries[i]);
        if(fDiff > 0.0001f)
        {
            bRet = false;
            break;
        }
    }
    
    return bRet;
}

/*
**
*/
vec3 extractEulerAngles(mat4 const& m)
{
	vec3 ret;

	float const fTwoPI = 2.0f * (float)M_PI;

	float fSY = sqrtf(m.mafEntries[0] * m.mafEntries[0] + m.mafEntries[4] * m.mafEntries[4]);   // (0,0), (1,0)

	if(fSY < 1e-6)
	{
		ret.x = atan2f(-m.mafEntries[6], m.mafEntries[5]);     // (1,2) (1,1)
		ret.y = atan2f(-m.mafEntries[8], fSY);      // (2,0)
		ret.z = 0.0f;
}
	else
	{
		ret.x = atan2f(m.mafEntries[9], m.mafEntries[10]);    // (2,1) (2,2)
		ret.y = atan2f(-m.mafEntries[8], fSY);     // (2,0)
		ret.z = atan2f(m.mafEntries[4], m.mafEntries[0]);     // (1,0) (0,0)
	}

	return ret;
}

/*
**
*/
bool mat4::identical(mat4 const& m, float fTolerance) const
{
	bool bRet = true;
	for(uint32_t i = 0; i < 16; i++)
	{
		float fDiff = fabsf(mafEntries[i] - m.mafEntries[i]);
		if(fDiff > fTolerance)
		{
			bRet = false;
			break;
		}
	}

	return bRet;
}

/*
**
*/
mat4 makeFromAngleAxis(vec3 const& axis, float fAngle)
{
    float fCosAngle = cosf(fAngle);
    float fSinAngle = sinf(fAngle);
    float fT = 1.0f - fCosAngle;

    mat4 m;
    m.mafEntries[0] = fT * axis.x * axis.x + fCosAngle;
    m.mafEntries[5] = fT * axis.y * axis.y + fCosAngle;
    m.mafEntries[10] = fT * axis.z * axis.z + fCosAngle;

    float fTemp0 = axis.x * axis.y * fT;
    float fTemp1 = axis.z * fSinAngle;

    m.mafEntries[4] = fTemp0 + fTemp1;
    m.mafEntries[1] = fTemp0 - fTemp1;

    fTemp0 = axis.x * axis.z * fT;
    fTemp1 = axis.y * fSinAngle;

    m.mafEntries[8] = fTemp0 - fTemp1;
    m.mafEntries[2] = fTemp0 + fTemp1;

    fTemp0 = axis.y * axis.z * fT;
    fTemp1 = axis.x * fSinAngle;

    m.mafEntries[9] = fTemp0 + fTemp1;
    m.mafEntries[6] = fTemp0 - fTemp1;
    

    return m;
}

/*
**
*/
void makeAngleAxis(
    float3& axis,
    float& fAngle,
    mat4 const& R)
{
    float fTrace = R.mafEntries[0] + R.mafEntries[5] + R.mafEntries[10];
    float fVal = maxf(minf((fTrace - 1.0f) * 0.5f, 1.0f), -1.0f);
    fAngle = acosf(fVal);
    if(fAngle == 0.0f)
    {
        axis = float3(1.0f, 0.0f, 0.0f);
    }
    else
    {
        axis = normalize(float3(R.mafEntries[9] - R.mafEntries[6], R.mafEntries[2] - R.mafEntries[8], R.mafEntries[4] - R.mafEntries[1]) / (2.0f * sinf(fAngle)));
    }
}

/*
**
*/
mat4 makeRotation(
    vec3 const& dest, 
    vec3 const& src)
{
    // axis and angle of rotation
    float3 axis = cross(src, dest);
    float3 axisNormalized = normalize(axis);
    float fAnimAngle = atan2f(length(axis), dot(dest, src));

    // K matrix for Rodriguez rotation
    float4x4 K;
    K.mafEntries[0] = 0.0f;
    K.mafEntries[1] = -axisNormalized.z;
    K.mafEntries[2] = axisNormalized.y;
    K.mafEntries[3] = 0.0f;

    K.mafEntries[4] = axisNormalized.z;
    K.mafEntries[5] = 0.0f;
    K.mafEntries[6] = -axisNormalized.x;
    K.mafEntries[7] = 0.0f;

    K.mafEntries[8] = -axisNormalized.y;
    K.mafEntries[9] = axisNormalized.x;
    K.mafEntries[10] = 0.0f;
    K.mafEntries[11] = 0.0f;

    K.mafEntries[12] = 0.0f;
    K.mafEntries[13] = 0.0f;
    K.mafEntries[14] = 0.0f;
    K.mafEntries[15] = 1.0f;

    float fSinAngle = sinf(fAnimAngle);
    float fOneMinusCosAngle = 1.0f - cosf(fAnimAngle);

    // Rodriguez rotation matrix
    float4x4 R;
    float4x4 I;
    float afEntries[16];
    afEntries[0] = I.mafEntries[0] + fSinAngle * K.mafEntries[0] + fOneMinusCosAngle * K.mafEntries[0] * K.mafEntries[0];
    afEntries[1] = I.mafEntries[1] + fSinAngle * K.mafEntries[1] + fOneMinusCosAngle * K.mafEntries[1] * K.mafEntries[1];
    afEntries[2] = I.mafEntries[2] + fSinAngle * K.mafEntries[2] + fOneMinusCosAngle * K.mafEntries[2] * K.mafEntries[2];
    afEntries[3] = I.mafEntries[3] + fSinAngle * K.mafEntries[3] + fOneMinusCosAngle * K.mafEntries[3] * K.mafEntries[3];

    afEntries[4] = I.mafEntries[4] + fSinAngle * K.mafEntries[4] + fOneMinusCosAngle * K.mafEntries[4] * K.mafEntries[4];
    afEntries[5] = I.mafEntries[5] + fSinAngle * K.mafEntries[5] + fOneMinusCosAngle * K.mafEntries[5] * K.mafEntries[5];
    afEntries[6] = I.mafEntries[6] + fSinAngle * K.mafEntries[6] + fOneMinusCosAngle * K.mafEntries[6] * K.mafEntries[6];
    afEntries[7] = I.mafEntries[7] + fSinAngle * K.mafEntries[7] + fOneMinusCosAngle * K.mafEntries[7] * K.mafEntries[7];

    afEntries[8] = I.mafEntries[8] + fSinAngle * K.mafEntries[8] + fOneMinusCosAngle * K.mafEntries[8] * K.mafEntries[8];
    afEntries[9] = I.mafEntries[9] + fSinAngle * K.mafEntries[9] + fOneMinusCosAngle * K.mafEntries[9] * K.mafEntries[9];
    afEntries[10] = I.mafEntries[10] + fSinAngle * K.mafEntries[10] + fOneMinusCosAngle * K.mafEntries[10] * K.mafEntries[10];
    afEntries[11] = I.mafEntries[11] + fSinAngle * K.mafEntries[11] + fOneMinusCosAngle * K.mafEntries[11] * K.mafEntries[11];

    afEntries[12] = I.mafEntries[12] + fSinAngle * K.mafEntries[12] + fOneMinusCosAngle * K.mafEntries[12] * K.mafEntries[12];
    afEntries[13] = I.mafEntries[13] + fSinAngle * K.mafEntries[13] + fOneMinusCosAngle * K.mafEntries[13] * K.mafEntries[13];
    afEntries[14] = I.mafEntries[14] + fSinAngle * K.mafEntries[14] + fOneMinusCosAngle * K.mafEntries[14] * K.mafEntries[14];
    afEntries[15] = I.mafEntries[15] + fSinAngle * K.mafEntries[15] + fOneMinusCosAngle * K.mafEntries[15] * K.mafEntries[15];

    float3 v0 = normalize(float3(afEntries[0], afEntries[1], afEntries[2]));
    float3 v1 = normalize(float3(afEntries[4], afEntries[5], afEntries[6]));
    float3 v2 = normalize(float3(afEntries[8], afEntries[9], afEntries[10]));
    
    return float4x4(v0, v1, v2);
}