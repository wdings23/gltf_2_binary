#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <vector>
#include <string>
#include <map>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tinygltf/tiny_gltf.h>

#include <math/vec.h>
#include <math/mat4.h>
#include <math/quaternion.h>

#include <utils/LogPrint.h>

struct AnimFrame
{
    float           mfTime;
    uint32_t        miNodeIndex;
    float4          mRotation;
    float4          mTranslation;
    float4          mScaling;
};

struct Joint
{
    uint32_t miIndex;
    uint32_t miNumChildren;
    uint32_t maiChildren[32];
    uint32_t miParent;

    quaternion  mRotation;
    float3      mTranslation;
    float3      mScaling;
};

void traverseRig(
    std::map<uint32_t, float4x4>& aGlobalJointPositions,
    Joint const& joint,
    float4x4 const& matrix,
    std::vector<Joint> const& aJoints,
    std::vector<std::vector<AnimFrame>> const& aaKeyFrames,
    std::vector<uint32_t> const& aiSkinJointIndices,
    std::vector<float4x4> const& aJointLocalBindMatrices,
    std::vector<float4x4> const& aJointGlobalInverseBindMatrices,
    std::vector<uint32_t> const& aiJointIndexMapping,
    std::map<uint32_t, std::string>& aJointMapping,
    uint32_t iStack);

/*
**
*/
void loadGLTF(
    std::string const& filePath,
    std::string const& outputFilePath)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filePath.c_str());
    assert(ret);

    std::vector<std::vector<float3>> aaMeshPositions;
    std::vector<std::vector<float3>> aaMeshNormals;
    std::vector<std::vector<float2>> aaMeshTexCoords;
    std::vector<std::vector<uint32_t>> aaiMeshTriangleIndices;

    {
        std::vector<std::string> aAttributeTypes;
        aAttributeTypes.push_back("POSITION");
        aAttributeTypes.push_back("NORMAL");
        aAttributeTypes.push_back("TEXCOORD_0");

        for(const auto& mesh : model.meshes)
        {
            DEBUG_PRINTF("mesh: %s primitives: %d\n", mesh.name.c_str(), mesh.primitives.size());

            for(const auto& primitive : mesh.primitives)
            {
                // indices
                std::vector<uint32_t> aiMeshTriangleIndices;
                {
                    const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                    const tinygltf::BufferView& indexView = model.bufferViews[indexAccessor.bufferView];
                    const tinygltf::Buffer& indexBuffer = model.buffers[indexView.buffer];

                    const unsigned char* bufferStart = indexBuffer.data.data() + indexView.byteOffset + indexAccessor.byteOffset;
                    aiMeshTriangleIndices.resize(indexAccessor.count);
                    for(size_t i = 0; i < indexAccessor.count; ++i)
                    {
                        uint32_t index = 0;
                        switch(indexAccessor.componentType) 
                        {
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                                index = *(reinterpret_cast<const uint8_t*>(bufferStart + i));
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                                index = *(reinterpret_cast<const uint16_t*>(bufferStart + i * sizeof(uint16_t)));
                                break;
                            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                                index = *(reinterpret_cast<const uint32_t*>(bufferStart + i * sizeof(uint32_t)));
                                break;
                            default:
                                assert(0);
                                return;
                        }

                        aiMeshTriangleIndices[i] = index;
                    }

                    aaiMeshTriangleIndices.push_back(aiMeshTriangleIndices);
                }

                // attributes
                for(auto const& attributeType : aAttributeTypes)
                {
                    auto attr = primitive.attributes.find(attributeType);
                    if(attr == primitive.attributes.end())
                    {
                        continue;
                    }

                    int accessorIndex = attr->second;
                    const tinygltf::Accessor& accessor = model.accessors[accessorIndex];
                    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                    size_t byteOffset = bufferView.byteOffset + accessor.byteOffset;
                    const unsigned char* dataPtr = buffer.data.data() + byteOffset;

                    DEBUG_PRINTF("\tvertex count: %d\n", accessor.count);

                    // Assume float3 (VEC3, float)
                    if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
                        accessor.type == TINYGLTF_TYPE_VEC3)
                    {
                        uint32_t iNumPositions = (uint32_t)(bufferView.byteLength / sizeof(float3));
                        assert(bufferView.byteLength % sizeof(float3) == 0);
                        assert(iNumPositions == accessor.count);
                        std::vector<float3> aMeshPositions(iNumPositions);

                        memcpy(
                            aMeshPositions.data(),
                            dataPtr,
                            bufferView.byteLength);

                        if(attributeType == "POSITION")
                        {
                            aaMeshPositions.push_back(aMeshPositions);
                        }
                        else if(attributeType == "NORMAL")
                        {
                            aaMeshNormals.push_back(aMeshPositions);
                        }
                    }
                    else if(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT &&
                        accessor.type == TINYGLTF_TYPE_VEC2)
                    {
                        uint32_t iNumPositions = (uint32_t)(bufferView.byteLength / sizeof(float2));
                        assert(bufferView.byteLength % sizeof(float2) == 0);
                        assert(iNumPositions == accessor.count);
                        std::vector<float2> aMeshPositions(iNumPositions);

                        memcpy(
                            aMeshPositions.data(),
                            dataPtr,
                            bufferView.byteLength);

                        if(attributeType == "TEXCOORD_0")
                        {
                            aaMeshTexCoords.push_back(aMeshPositions);
                        }
                    }

                }   // for attribute

            }   // for primitive

        }   // for mesh

    }   // load mesh attributes

    // load skinning attributes
    std::vector<std::vector<Joint>> aaJoints;
    std::vector<std::vector<uint32_t>> aaaiJointInfluences;
    std::vector<std::vector<float>> aaafJointWeights;
    std::vector<std::vector<float4x4>> aaInverseBindMatrices;
    std::vector<std::vector<float4x4>> aaBindMatrices;
    std::vector<std::vector<uint32_t>> aaiSkinJointIndices;
    std::vector<std::vector<Joint>> aaAnimHierarchy;
    std::vector<std::vector<float4x4>> aaJointLocalBindMatrices;
    std::vector<std::vector<AnimFrame>> aaAnimationFrames;
    std::vector<std::vector<uint32_t>> aaiJointMapIndices;
    std::map<std::string, std::map<uint32_t, std::string>> aaJointMapping;
    {
        for(const auto& mesh : model.meshes)
        {
            DEBUG_PRINTF("mesh: %s primitives: %d\n", mesh.name.c_str(), mesh.primitives.size());

            for(const auto& primitive : mesh.primitives)
            {
                // attributes
                auto jointAttr = primitive.attributes.find("JOINTS_0");
                if(jointAttr == primitive.attributes.end())
                {
                    continue;
                }

                auto weightAttr = primitive.attributes.find("WEIGHTS_0");
                if(weightAttr == primitive.attributes.end())
                {
                    continue;
                }

                const tinygltf::Accessor& jointsAccessor = model.accessors[jointAttr->second];
                const tinygltf::Accessor& weightsAccessor = model.accessors[weightAttr->second];

                const tinygltf::BufferView& jointsView = model.bufferViews[jointsAccessor.bufferView];
                const tinygltf::BufferView& weightsView = model.bufferViews[weightsAccessor.bufferView];

                const tinygltf::Buffer& jointsBuffer = model.buffers[jointsView.buffer];
                const tinygltf::Buffer& weightsBuffer = model.buffers[weightsView.buffer];

                const uint8_t* jointsData = jointsBuffer.data.data() + jointsView.byteOffset + jointsAccessor.byteOffset;
                const uint8_t* weightsData = weightsBuffer.data.data() + weightsView.byteOffset + weightsAccessor.byteOffset;

                size_t count = jointsAccessor.count;

                std::vector<float> aafVertexWeights;
                std::vector<uint32_t> aaiJointInfluences;
                for(size_t i = 0; i < count; ++i)
                {
                    if(jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                    {
                        std::vector<uint32_t> aiJointInfluences(4);
                        std::vector<float> afJointWeights(4);

                        uint8_t const* joint = (uint8_t const*)(jointsData + i * 4); // Typically UBYTE or USHORT
                        const float* weight = (const float*)(weightsData + i * 4 * sizeof(float));

                        aaiJointInfluences.push_back((uint32_t)joint[0]);
                        aaiJointInfluences.push_back((uint32_t)joint[1]);
                        aaiJointInfluences.push_back((uint32_t)joint[2]);
                        aaiJointInfluences.push_back((uint32_t)joint[3]);

                        aafVertexWeights.push_back(weight[0]);
                        aafVertexWeights.push_back(weight[1]);
                        aafVertexWeights.push_back(weight[2]);
                        aafVertexWeights.push_back(weight[3]);

                        DEBUG_PRINTF("v %d joint: [%d, %d, %d, %d] weights: [%.4f, %.4f, %.4f, %.4f]\n",
                            i,
                            joint[0], joint[1], joint[2], joint[3],
                            weight[0], weight[1], weight[2], weight[3]);
                    }
                    else if(jointsAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                    {
                        std::vector<uint32_t> aiJointInfluences(4);
                        std::vector<float> afJointWeights(4);

                        uint16_t const* joint = (uint16_t const*)(jointsData + i * 4); // Typically UBYTE or USHORT
                        const float* weight = (const float*)(weightsData + i * 4 * sizeof(float));

                        aaiJointInfluences.push_back((uint32_t)joint[0]);
                        aaiJointInfluences.push_back((uint32_t)joint[1]);
                        aaiJointInfluences.push_back((uint32_t)joint[2]);
                        aaiJointInfluences.push_back((uint32_t)joint[3]);

                        aafVertexWeights.push_back(weight[0]);
                        aafVertexWeights.push_back(weight[1]);
                        aafVertexWeights.push_back(weight[2]);
                        aafVertexWeights.push_back(weight[3]);

                        DEBUG_PRINTF("v %d joint: [%d, %d, %d, %d] weights: [%.4f, %.4f, %.4f, %.4f]\n",
                            i,
                            joint[0], joint[1], joint[2], joint[3],
                            weight[0], weight[1], weight[2], weight[3]);
                    }

                }   // for i = 0 to count

                aaaiJointInfluences.push_back(aaiJointInfluences);
                aaafJointWeights.push_back(aafVertexWeights);
            }

        }   // for mesh

        // skin hiererachy
        {
            for(size_t s = 0; s < model.skins.size(); ++s) 
            {
                std::vector<Joint> aJoints;

                const tinygltf::Skin& skin = model.skins[s];
                DEBUG_PRINTF("skin: %s\n", skin.name.c_str());

                // Optional root node of the skeleton
                if(skin.skeleton >= 0 && skin.skeleton < model.nodes.size())
                {
                    DEBUG_PRINTF("skeleton root: %s\n", model.nodes[skin.skeleton].name.c_str());
                }

                // All joint node indices
                for(int jointIndex : skin.joints) 
                {
                    Joint joint;
                    if(jointIndex < 0 || jointIndex >= model.nodes.size())
                    {
                        continue;
                    }

                    const auto& jointNode = model.nodes[jointIndex];
                    DEBUG_PRINTF("\tjoint %s index %d\n", jointNode.name.c_str(), jointIndex);

                    
                    joint.mRotation = (jointNode.rotation.size() > 0) ? quaternion((float)jointNode.rotation[0], (float)jointNode.rotation[1], (float)jointNode.rotation[2], (float)jointNode.rotation[3]) : quaternion(0.0f, 0.0f, 0.0f, 1.0f);
                    joint.mTranslation = (jointNode.translation.size() > 0) ? float3((float)jointNode.translation[0], (float)jointNode.translation[1], (float)jointNode.translation[2]) : float3(0.0f, 0.0f, 0.0f);
                    joint.mScaling = (jointNode.scale.size() > 0) ? float3((float)jointNode.scale[0], (float)jointNode.scale[1], (float)jointNode.scale[2]) : float3(1.0f, 1.0f, 1.0f);

                    joint.miIndex = jointIndex;
                    joint.miNumChildren = 0;
                    memset(joint.maiChildren, 0xff, sizeof(joint.maiChildren));
                    for(int child : jointNode.children) 
                    {
                        const auto& childNode = model.nodes[child];
                        
                        joint.maiChildren[joint.miNumChildren++] = child;
                        
                        DEBUG_PRINTF("\t\tchild %s index %d\n", childNode.name.c_str(), child);
                    }

                    aJoints.push_back(joint);
                }

                // tag joints that are children of some other joints
                std::vector<uint32_t> aiTagged(aJoints.size());
                std::vector<Joint> aRootJoints;
                for(auto const& joint : aJoints)
                {
                    for(uint32_t i = 0; i < joint.miNumChildren; i++)
                    {
                        aiTagged[joint.maiChildren[i]] = 1;
                    }
                }
                
                // get the root of the hierarchy
                for(uint32_t i = 0; i < (uint32_t)aJoints.size(); i++)
                {
                    if(aiTagged[i] == 0)
                    {
                        for(uint32_t j = 0; j < (uint32_t)aJoints.size(); j++)
                        {
                            if(aJoints[j].miIndex == i)
                            {
                                aRootJoints.push_back(aJoints[j]);
                                break;
                            }
                        }
                    }
                }

                std::vector<float4x4> aJointLocalBindMatrices;
                for(auto const& joint : aJoints)
                {
                    float4x4 rotationMatrix = joint.mRotation.matrix();
                    float4x4 translationMatrix = translate(joint.mTranslation.x, joint.mTranslation.y, joint.mTranslation.z);
                    float4x4 scaleMatrix = scale(joint.mScaling.x, joint.mScaling.y, joint.mScaling.z);

                    float4x4 localMatrix = translationMatrix * rotationMatrix * scaleMatrix;
                    aJointLocalBindMatrices.push_back(localMatrix);
                }

                aaJoints.push_back(aJoints);
                aaAnimHierarchy.push_back(aRootJoints);
                aaJointLocalBindMatrices.push_back(aJointLocalBindMatrices);
            }
        }

        // skin
        for(const auto& skin : model.skins)
        {
            DEBUG_PRINTF("skin name: %s num joints: %d\n", skin.name.c_str(), skin.joints.size());

            const tinygltf::Accessor& ibmAccessor = model.accessors[skin.inverseBindMatrices];
            const tinygltf::BufferView& ibmView = model.bufferViews[ibmAccessor.bufferView];
            const tinygltf::Buffer& ibmBuffer = model.buffers[ibmView.buffer];

            const float* ibmData = reinterpret_cast<const float*>(
                ibmBuffer.data.data() + ibmView.byteOffset + ibmAccessor.byteOffset);

            size_t jointCount = skin.joints.size();
            std::vector<float4x4> aInverseBindMatrices(jointCount);
            memcpy(
                aInverseBindMatrices.data(),
                ibmData,
                sizeof(float4x4) * jointCount);
            aaInverseBindMatrices.push_back(aInverseBindMatrices);

            std::vector<float4x4> aBindMatrices;
            for(uint32_t i = 0; i < aInverseBindMatrices.size(); i++)
            {
                float4x4 bindMatrix = invert(aInverseBindMatrices[i]);
                aBindMatrices.push_back(bindMatrix);
            }
            aaBindMatrices.push_back(aBindMatrices);

            std::vector<uint32_t> aiJointIndices(jointCount);
            memcpy(
                aiJointIndices.data(),
                skin.joints.data(),
                jointCount * sizeof(uint32_t)
            );
            aaiSkinJointIndices.push_back(aiJointIndices);

            for(uint32_t i = 0; i < aiJointIndices.size(); i++)
            {
                uint32_t iJointIndex = aiJointIndices[i];
                aaJointMapping[skin.name][iJointIndex] = model.nodes[iJointIndex].name;
            }

            for(auto const& keyValue : aaJointMapping[skin.name])
            {
                DEBUG_PRINTF("\tkey: %d value: %s\n", keyValue.first, keyValue.second.c_str());
            }

        }   // for skin

        for(uint32_t i = 0; i < (uint32_t)aaiSkinJointIndices.size(); i++)
        {
            std::vector<uint32_t> aiJointMapIndices(aaiSkinJointIndices[i].size());
            for(uint32_t j = 0; j < (uint32_t)aaiSkinJointIndices[i].size(); j++)
            {
                uint32_t iJointIndex = aaiSkinJointIndices[i][j];
                aiJointMapIndices[iJointIndex] = j;
            }

            aaiJointMapIndices.push_back(aiJointMapIndices);
        }

        for(uint32_t i = 0; i < (uint32_t)aaJoints.size(); i++)
        {
            for(auto const& joint : aaJoints[i])
            {
                for(uint32_t j = 0; j < joint.miNumChildren; j++)
                {
                    uint32_t iArrayIndex = aaiJointMapIndices[0][joint.maiChildren[j]];
                    aaJoints[i][iArrayIndex].miParent = joint.miIndex;
                }
            }
        }

    }   // skinning attributes

    // animation
    {
        std::map<std::string, uint32_t> aNodeToJointNameMappings;
        for(size_t i = 0; i < model.animations.size(); ++i) 
        {
            const tinygltf::Animation& anim = model.animations[i];
            DEBUG_PRINTF("animation %d name: %s\n", i, anim.name.c_str());

            std::vector<float4> aTranslations;
            std::vector<float4> aRotations;
            std::vector<float4> aScalings;
            std::vector<std::vector<float>> aafTime;
            std::vector<uint32_t> aiNodeIndices;
            std::vector<std::string> aNodeNames;
            uint32_t iChannel = 0;
            for(const auto& channel : anim.channels) 
            {
                const tinygltf::AnimationSampler& sampler = anim.samplers[channel.sampler];
                const std::string& path = channel.target_path; // "translation", "rotation", "scale", or "weights"
                int nodeIndex = channel.target_node;
                const std::string& nodeName = (nodeIndex >= 0 && nodeIndex < model.nodes.size()) ? model.nodes[nodeIndex].name : "<unnamed>";

                aNodeToJointNameMappings[nodeName] = nodeIndex;
                DEBUG_PRINTF("\tnode: %s index: %d\n", nodeName.c_str(), nodeIndex);

                // Load input times
                const auto& inputAccessor = model.accessors[sampler.input];
                const auto& inputBufferView = model.bufferViews[inputAccessor.bufferView];
                const auto& inputBuffer = model.buffers[inputBufferView.buffer];
                const float* timeData = reinterpret_cast<const float*>(
                    inputBuffer.data.data() + inputBufferView.byteOffset + inputAccessor.byteOffset);

                // Load output values
                const auto& outputAccessor = model.accessors[sampler.output];
                const auto& outputBufferView = model.bufferViews[outputAccessor.bufferView];
                const auto& outputBuffer = model.buffers[outputBufferView.buffer];
                const float* valueData = reinterpret_cast<const float*>(
                    outputBuffer.data.data() + outputBufferView.byteOffset + outputAccessor.byteOffset);

                DEBUG_PRINTF("\t\tnum key frames: %d\n", inputAccessor.count);

                // channel data
                std::vector<float> afTime;
                for(size_t j = 0; j < inputAccessor.count; ++j) 
                {
                    afTime.push_back(timeData[j]);
                    aNodeNames.push_back(nodeName);
                    aiNodeIndices.push_back(nodeIndex);

                    if(path == "translation") 
                    {
                        aTranslations.push_back(float4(valueData[j * 3], valueData[j * 3 + 1], valueData[j * 3 + 2], 0.0f));
                        DEBUG_PRINTF("\t\t\ttime: %.4f %s (%.4f, %.4f, %4f)\n", 
                            timeData[j],
                            path.c_str(),
                            valueData[j * 3], 
                            valueData[j * 3 + 1], 
                            valueData[j * 3 + 2]);
                    }
                    else if(path == "scale")
                    {
                        aScalings.push_back(float4(valueData[j * 3], valueData[j * 3 + 1], valueData[j * 3 + 2], 0.0f));
                        DEBUG_PRINTF("\t\t\ttime: %.4f %s (%.4f, %.4f, %4f)\n",
                            timeData[j],
                            path.c_str(),
                            valueData[j * 3],
                            valueData[j * 3 + 1],
                            valueData[j * 3 + 2]);
                    }
                    else if(path == "rotation") 
                    {
                        aRotations.push_back(float4(valueData[j * 4], valueData[j * 4 + 1], valueData[j * 4 + 2], valueData[j * 4 + 3]));
                        DEBUG_PRINTF("\t\t\ttime: %.4f %s (%.4f, %.4f, %4f, %.4f)\n", 
                            timeData[j],
                            path.c_str(),
                            valueData[j * 4], 
                            valueData[j * 4 + 1], 
                            valueData[j * 4 + 2], 
                            valueData[j * 4 + 3]);
                    }

                }   // for accessor
            
                ++iChannel;

                if(iChannel > 0 && iChannel % 3 == 0)
                {
                    assert(aRotations.size() == aTranslations.size() && aTranslations.size() == aScalings.size());
                    aafTime.push_back(afTime);

                    std::vector<AnimFrame> aAnimationFrames(aRotations.size());
                    for(uint32_t i = 0; i < aRotations.size(); i++)
                    {
                        AnimFrame& animationFrame = aAnimationFrames[i];

                        animationFrame.mRotation = aRotations[i];
                        animationFrame.mTranslation = aTranslations[i];
                        animationFrame.mScaling = aScalings[i];
                        animationFrame.mfTime = afTime[i];
                        animationFrame.miNodeIndex = aiNodeIndices[i];
                    }

                    aaAnimationFrames.push_back(aAnimationFrames);

                    aTranslations.clear();
                    aScalings.clear();
                    aRotations.clear();
                    afTime.clear();
                    aiNodeIndices.clear();
                }

            }   // for channel

        }   // for animation

    }   // animation 

    uint32_t iNumMeshes = (uint32_t)aaMeshPositions.size();
    uint32_t iNumAnimations = (uint32_t)aaInverseBindMatrices.size();
   
    // save out mesh data
    FILE* fp = fopen(outputFilePath.c_str(), "wb");
    fwrite(&iNumMeshes, sizeof(uint32_t), iNumMeshes, fp);
    fwrite(&iNumAnimations, sizeof(uint32_t), iNumAnimations, fp);
    for(uint32_t i = 0; i < aaMeshPositions.size(); i++)
    {
        std::vector<float3> const& aMeshPositions = aaMeshPositions[i];
        uint32_t iNumMeshPositions = (uint32_t)aMeshPositions.size();
        fwrite(&iNumMeshPositions, sizeof(uint32_t), 1, fp);
        fwrite(aMeshPositions.data(), sizeof(float3), iNumMeshPositions, fp);

        std::vector<float3> const& aMeshNormals = aaMeshNormals[i];
        uint32_t iNumMeshNormals = (uint32_t)aMeshNormals.size();
        fwrite(&iNumMeshNormals, sizeof(uint32_t), 1, fp);
        fwrite(aMeshNormals.data(), sizeof(float3), iNumMeshNormals, fp);

        std::vector<float2> const& aMeshTexCoords = aaMeshTexCoords[i];
        uint32_t iNumMeshTexCoords = (uint32_t)aaMeshTexCoords.size();
        fwrite(&iNumMeshTexCoords, sizeof(uint32_t), 1, fp);
        fwrite(aMeshTexCoords.data(), sizeof(float2), iNumMeshTexCoords, fp);

        std::vector<uint32_t> const& aiMeshTriangleIndices = aaiMeshTriangleIndices[i];
        uint32_t iNumTriangleIndices = (uint32_t)aiMeshTriangleIndices.size();
        fwrite(&iNumTriangleIndices, sizeof(uint32_t), 1, fp);
        fwrite(aiMeshTriangleIndices.data(), sizeof(uint32_t), iNumTriangleIndices, fp);
    }

    // save out animations
    for(uint32_t i = 0; i < iNumAnimations; i++)
    {
        uint32_t iNumTotalJointWeights = (uint32_t)aaaiJointInfluences[i].size();
        std::vector<uint32_t> const& aaiJointInfluences = aaaiJointInfluences[i];
        std::vector<float> const& aafJointInfluences = aaafJointWeights[i];

        assert(iNumTotalJointWeights == aaMeshPositions[i].size() * 4);

        fwrite(&iNumTotalJointWeights, sizeof(uint32_t), 1, fp);
        fwrite(aaiJointInfluences.data(), sizeof(uint32_t), iNumTotalJointWeights, fp);
        fwrite(&iNumTotalJointWeights, sizeof(uint32_t), 1, fp);
        fwrite(aafJointInfluences.data(), sizeof(float), iNumTotalJointWeights, fp);

        std::vector<uint32_t> const& aiSkinJointIndices = aaiSkinJointIndices[i];
        uint32_t iNumJoints = (uint32_t)aiSkinJointIndices.size();
        fwrite(&iNumJoints, sizeof(uint32_t), 1, fp);
        fwrite(aiSkinJointIndices.data(), sizeof(uint32_t), iNumJoints, fp);

        std::vector<float4x4> const& aInverseBindMatrices = aaInverseBindMatrices[i];
        iNumJoints = (uint32_t)aInverseBindMatrices.size();
        fwrite(&iNumJoints, sizeof(uint32_t), 1, fp);
        fwrite(aInverseBindMatrices.data(), sizeof(float4x4), iNumJoints, fp);
    }
    
    // animation hierarchy
    uint32_t iNumAnimHierarchies = (uint32_t)aaJoints.size();
    fwrite(&iNumAnimHierarchies, sizeof(uint32_t), 1, fp);
    for(uint32_t i = 0; i < (uint32_t)aaJoints.size(); i++)
    {
        std::vector<Joint> const& aJoints = aaJoints[i];
        uint32_t iNumJoints = (uint32_t)aJoints.size();
        fwrite(&iNumJoints, sizeof(uint32_t), 1, fp);
        fwrite(aJoints.data(), sizeof(Joint), iNumJoints, fp);
    }

    // joint to array mapping
    for(uint32_t i = 0; i < (uint32_t)aaiJointMapIndices.size(); i++)
    {
        uint32_t iNumJoints = (uint32_t)aaiJointMapIndices[i].size();
        fwrite(&iNumJoints, sizeof(uint32_t), 1, fp);
        fwrite(aaiJointMapIndices[i].data(), sizeof(uint32_t), iNumJoints, fp);
    }

    // save out joint mapping
    for(auto const& aJointMappingKeyValue : aaJointMapping)
    {
        uint32_t iNumJoints = (uint32_t)aJointMappingKeyValue.second.size();
        fwrite(&iNumJoints, sizeof(uint32_t), 1, fp);

        for(auto const& keyValue : aJointMappingKeyValue.second)
        {
            fwrite(&keyValue.first, sizeof(uint32_t), 1, fp);
            uint32_t iLength = (uint32_t)keyValue.second.length();
            fwrite(&iLength, sizeof(uint32_t), 1, fp);
            for(uint32_t j = 0; j < iLength; j++)
            {
                fwrite(&keyValue.second.at(j), sizeof(char), 1, fp);
            }
        }
    }

    fclose(fp);

    //Joint rootJoint = aaAnimHierarchy[0].front();
    //float4x4 const* pMatrix = &aaInverseBindMatrices[0][rootJoint.miIndex];
    //float4x4 rootMatrix = invert(transpose(*pMatrix));
    
    std::map<uint32_t, float4x4> aGlobalJointAnimatedMatrices;
    float4x4 rootMatrix;
    traverseRig(
        aGlobalJointAnimatedMatrices,
        aaAnimHierarchy[0].front(),
        rootMatrix,
        aaJoints[0],
        aaAnimationFrames,
        aaiSkinJointIndices[0],
        aaJointLocalBindMatrices[0],
        aaInverseBindMatrices[0],
        aaiJointMapIndices[0],
        aaJointMapping["Armature"],
        0
    );

    std::map<uint32_t, float4x4> aGlobalJointBindMatrices;
    for(uint32_t i = 0; i < (uint32_t)aaInverseBindMatrices[0].size(); i++)
    {
        Joint const& joint = aaJoints[0][i];
        float4x4 m = invert(transpose(aaInverseBindMatrices[0][i]));
        aGlobalJointBindMatrices[joint.miIndex] = m;
    }

    // test rotation conversion of bind root to spin0 joint in bind pose to animated pose
    {
        uint32_t iRootJointIndex = 0;
        uint32_t iRootArrayIndex = aaiJointMapIndices[0][iRootJointIndex];
        Joint const& rootJoint = aaJoints[0][0];
        uint32_t iChildJointIndex = rootJoint.maiChildren[0];
        uint32_t iChildArrayIndex = aaiJointMapIndices[0][iChildJointIndex];
        Joint const& childJoint = aaJoints[0][iChildArrayIndex];
        
        // animated joint position
        float3 rootJointAnimatedPosition = float3(
            aGlobalJointAnimatedMatrices[rootJoint.miIndex].mafEntries[3],
            aGlobalJointAnimatedMatrices[rootJoint.miIndex].mafEntries[7],
            aGlobalJointAnimatedMatrices[rootJoint.miIndex].mafEntries[11]
        );
        float3 childJointAnimatedPosition = float3(
            aGlobalJointAnimatedMatrices[childJoint.miIndex].mafEntries[3],
            aGlobalJointAnimatedMatrices[childJoint.miIndex].mafEntries[7],
            aGlobalJointAnimatedMatrices[childJoint.miIndex].mafEntries[11]
        );

        // bind joint positions
        float3 rootJointBindPosition = float3(
            aGlobalJointBindMatrices[rootJoint.miIndex].mafEntries[3],
            aGlobalJointBindMatrices[rootJoint.miIndex].mafEntries[7],
            aGlobalJointBindMatrices[rootJoint.miIndex].mafEntries[11]
        );
        float3 childJointBindPosition = float3(
            aGlobalJointBindMatrices[childJoint.miIndex].mafEntries[3],
            aGlobalJointBindMatrices[childJoint.miIndex].mafEntries[7],
            aGlobalJointBindMatrices[childJoint.miIndex].mafEntries[11]
        );

        // root to spine joint vector
        float3 bindPositionDiff = childJointBindPosition - rootJointBindPosition;
        float3 animPositionDiff = childJointAnimatedPosition - rootJointAnimatedPosition;
        float3 bindPositionDiffNormalized = normalize(bindPositionDiff);
        float3 animPositionDiffNormalized = normalize(animPositionDiff);
        
        // axis and angle of rotation
        float3 axis = cross(bindPositionDiffNormalized, animPositionDiffNormalized);
        float3 axisNormalized = normalize(axis);
        float fAnimAngle = atan2f(length(axis), dot(animPositionDiffNormalized, bindPositionDiffNormalized));

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
        R.mafEntries[0] = I.mafEntries[0] + fSinAngle * K.mafEntries[0] + fOneMinusCosAngle * K.mafEntries[0] * K.mafEntries[0];
        R.mafEntries[1] = I.mafEntries[1] + fSinAngle * K.mafEntries[1] + fOneMinusCosAngle * K.mafEntries[1] * K.mafEntries[1];
        R.mafEntries[2] = I.mafEntries[2] + fSinAngle * K.mafEntries[2] + fOneMinusCosAngle * K.mafEntries[2] * K.mafEntries[2];
        R.mafEntries[3] = I.mafEntries[3] + fSinAngle * K.mafEntries[3] + fOneMinusCosAngle * K.mafEntries[3] * K.mafEntries[3];

        R.mafEntries[4] = I.mafEntries[4] + fSinAngle * K.mafEntries[4] + fOneMinusCosAngle * K.mafEntries[4] * K.mafEntries[4];
        R.mafEntries[5] = I.mafEntries[5] + fSinAngle * K.mafEntries[5] + fOneMinusCosAngle * K.mafEntries[5] * K.mafEntries[5];
        R.mafEntries[6] = I.mafEntries[6] + fSinAngle * K.mafEntries[6] + fOneMinusCosAngle * K.mafEntries[6] * K.mafEntries[6];
        R.mafEntries[7] = I.mafEntries[7] + fSinAngle * K.mafEntries[7] + fOneMinusCosAngle * K.mafEntries[7] * K.mafEntries[7];

        R.mafEntries[8] =  I.mafEntries[8] + fSinAngle * K.mafEntries[8] + fOneMinusCosAngle * K.mafEntries[8] * K.mafEntries[8];
        R.mafEntries[9] =  I.mafEntries[9] + fSinAngle * K.mafEntries[9] + fOneMinusCosAngle * K.mafEntries[9] * K.mafEntries[9];
        R.mafEntries[10] = I.mafEntries[10] + fSinAngle * K.mafEntries[10] + fOneMinusCosAngle * K.mafEntries[10] * K.mafEntries[10];
        R.mafEntries[11] = I.mafEntries[11] + fSinAngle * K.mafEntries[11] + fOneMinusCosAngle * K.mafEntries[11] * K.mafEntries[11];

        R.mafEntries[12] = I.mafEntries[12] + fSinAngle * K.mafEntries[12] + fOneMinusCosAngle * K.mafEntries[12] * K.mafEntries[12];
        R.mafEntries[13] = I.mafEntries[13] + fSinAngle * K.mafEntries[13] + fOneMinusCosAngle * K.mafEntries[13] * K.mafEntries[13];
        R.mafEntries[14] = I.mafEntries[14] + fSinAngle * K.mafEntries[14] + fOneMinusCosAngle * K.mafEntries[14] * K.mafEntries[14];
        R.mafEntries[15] = I.mafEntries[15] + fSinAngle * K.mafEntries[15] + fOneMinusCosAngle * K.mafEntries[15] * K.mafEntries[15];

        float4 test = R * float4(bindPositionDiffNormalized, 1.0f);

        int iDebug = 1;
    }

    int iDebug = 1;
}

/*
**
*/
void traverseRig(
    std::map<uint32_t, float4x4>& aGlobalJointPositions,
    Joint const& joint,
    float4x4 const& matrix,
    std::vector<Joint> const& aJoints,
    std::vector<std::vector<AnimFrame>> const& aaKeyFrames,
    std::vector<uint32_t> const& aiSkinJointIndices,
    std::vector<float4x4> const& aJointLocalBindMatrices,
    std::vector<float4x4> const& aJointGlobalInverseBindMatrices,
    std::vector<uint32_t> const& aiJointIndexMapping,
    std::map<uint32_t, std::string>& aJointMapping,
    uint32_t iStack)
{
    std::string jointName = aJointMapping[joint.miIndex];

#if 0
    for(uint32_t i = 0; i < iStack; i++)
    {
        DEBUG_PRINTF("     ");
    }
    DEBUG_PRINTF("%s ", jointName.c_str());
    DEBUG_PRINTF("(%.4f, %.4f, %.4f)\n",
        matrix.mafEntries[3],
        matrix.mafEntries[7],
        matrix.mafEntries[11]
    );
#endif // #if 0

    float const fTime = 3.0f;

    uint32_t iArrayIndex = aiJointIndexMapping[joint.miIndex];
    std::vector<AnimFrame> const& aKeyFrames = aaKeyFrames[iArrayIndex];

    AnimFrame keyFrame;
    {
        uint32_t iKeyFrame = 0, iPrevKeyFrame = 0;
        for(uint32_t i = 0; i < (uint32_t)aKeyFrames.size(); i++)
        {
            if(aKeyFrames[i].mfTime > fTime)
            {
                iKeyFrame = i;
                if(i > 0)
                {
                    iPrevKeyFrame = i - 1;
                }

                break;
            }
        }

        AnimFrame const& prevKeyFrame = aKeyFrames[iPrevKeyFrame];
        AnimFrame const& currKeyFrame = aKeyFrames[iKeyFrame];

        float fDuration = currKeyFrame.mfTime - prevKeyFrame.mfTime;
        float fPct = (fTime - prevKeyFrame.mfTime) / fDuration;
        keyFrame.mTranslation = prevKeyFrame.mTranslation + (currKeyFrame.mTranslation - prevKeyFrame.mTranslation) * fPct;
        keyFrame.mScaling = prevKeyFrame.mScaling + (currKeyFrame.mScaling - prevKeyFrame.mScaling) * fPct;
        keyFrame.mRotation.x = prevKeyFrame.mRotation.x + (currKeyFrame.mRotation.x - prevKeyFrame.mRotation.x) * fPct;
        keyFrame.mRotation.y = prevKeyFrame.mRotation.y + (currKeyFrame.mRotation.y - prevKeyFrame.mRotation.y) * fPct;
        keyFrame.mRotation.z = prevKeyFrame.mRotation.z + (currKeyFrame.mRotation.z - prevKeyFrame.mRotation.z) * fPct;
        keyFrame.mRotation.w = prevKeyFrame.mRotation.w + (currKeyFrame.mRotation.w - prevKeyFrame.mRotation.w) * fPct;

        int iDebug = 1;
    }

    float4x4 translateMatrix = translate(keyFrame.mTranslation);
    float4x4 scaleMatrix = scale(keyFrame.mScaling);
    quaternion q = quaternion(keyFrame.mRotation.x, keyFrame.mRotation.y, keyFrame.mRotation.z, keyFrame.mRotation.w);
    float4x4 rotationMatrix = q.matrix();
    float4x4 localAnimMatrix = translateMatrix * rotationMatrix * scaleMatrix;
    float4x4 totalMatrix = matrix * localAnimMatrix;

    aGlobalJointPositions[joint.miIndex] = totalMatrix;

    float4x4 globalBindMatrix = invert(transpose(aJointGlobalInverseBindMatrices[iArrayIndex]));

    //DEBUG_PRINTF("draw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0, 255, \"%s\")\n",
    //    totalMatrix.mafEntries[3],
    //    totalMatrix.mafEntries[7],
    //    totalMatrix.mafEntries[11],
    //    jointName.c_str()
    //);

    

    DEBUG_PRINTF("draw_sphere([%.4f, %.4f, %.4f], 0.01, 255, 0, 0, 255, \"%s\")\n",
        globalBindMatrix.mafEntries[3],
        globalBindMatrix.mafEntries[7],
        globalBindMatrix.mafEntries[11],
        jointName.c_str()
    );

    for(uint32_t i = 0; i < joint.miNumChildren; i++)
    {
        uint32_t iChildJointIndex = joint.maiChildren[i];
        uint32_t iChildArrayIndex = aiJointIndexMapping[iChildJointIndex];

        Joint const* pChildJoint = &aJoints[iChildArrayIndex];

        if(pChildJoint != nullptr)
        {
            traverseRig(
                aGlobalJointPositions,
                *pChildJoint,
                totalMatrix,
                aJoints,
                aaKeyFrames,
                aiSkinJointIndices,
                aJointLocalBindMatrices,
                aJointGlobalInverseBindMatrices,
                aiJointIndexMapping,
                aJointMapping,
                iStack + 1);
        }
    }
}

/*
**
*/
int main(int argc, char** argv)
{
    PrintOptions printOptions;
    printOptions.mbDisplayTime = false;
    DEBUG_PRINTF_SET_OPTIONS(printOptions);

    loadGLTF(argv[1], argv[2]);

    return 0;
}