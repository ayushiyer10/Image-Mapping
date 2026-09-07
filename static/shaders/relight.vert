varying vec2 vUv;
varying vec3 vWorldPosition;
varying vec3 vNormal;

uniform sampler2D uHeightMap;
uniform float uDisplacementScale;

void main() {
    vUv = uv;
    
    // Sample displacement height from grayscale height map texture
    float height = texture2D(uHeightMap, uv).r;
    vec3 displacedPosition = position + vec3(0.0, 0.0, height * uDisplacementScale);

    vNormal = normalMatrix * normal;
    vec4 worldPos = modelMatrix * vec4(displacedPosition, 1.0);
    vWorldPosition = worldPos.xyz;

    gl_Position = projectionMatrix * viewMatrix * worldPos;
}
