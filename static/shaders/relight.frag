varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vNormal;

uniform sampler2D uDiffuseMap;
uniform sampler2D uNormalMap;
uniform sampler2D uRoughnessMap;

uniform vec3 uLightPosition;
uniform vec3 uLightColor;
uniform float uAmbientIntensity;
uniform float uLightIntensity;
uniform float uSpecularStrength;
uniform float uRoughnessScale;

void main() {
    // 1. Sample textures
    vec3 albedo = texture2D(uDiffuseMap, vUv).rgb;
    vec3 normalTex = texture2D(uNormalMap, vUv).rgb;
    float roughTex = texture2D(uRoughnessMap, vUv).r * uRoughnessScale;

    // 2. Unpack surface normal from OpenGL RGB [0, 1] to [-1, 1]
    vec3 N = normalize(normalTex * 2.0 - 1.0);

    // 3. Vectors for lighting (Light L, View V, Halfway H)
    vec3 L = normalize(uLightPosition - vWorldPosition);
    vec3 V = normalize(cameraPosition - vWorldPosition);
    vec3 H = normalize(L + V);

    // 4. Ambient term
    vec3 ambient = uAmbientIntensity * albedo;

    // 5. Diffuse term (N dot L)
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = NdotL * albedo * uLightColor * uLightIntensity;

    // 6. Specular term (Blinn-Phong model parameterized by roughness)
    // Low roughness -> tight high-shininess specular highlight
    // High roughness -> broad muted specular highlight
    float shininess = mix(128.0, 4.0, clamp(roughTex, 0.0, 1.0));
    float NdotH = max(dot(N, H), 0.0);
    float specFactor = pow(NdotH, shininess);
    vec3 specular = specFactor * uSpecularStrength * (1.0 - roughTex * 0.6) * uLightColor * uLightIntensity;

    // 7. Composite color output
    vec3 color = ambient + diffuse + specular;

    gl_FragColor = vec4(color, 1.0);
}
